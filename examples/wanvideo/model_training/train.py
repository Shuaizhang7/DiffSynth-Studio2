import torch, os, argparse, accelerate, warnings
from tqdm import tqdm
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        import time
        os.makedirs(self.output_path, exist_ok=True)
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def load_training_state(self, accelerator, dir=None):
        if dir is None:
            return 0, 0
        meta_path = os.path.join(dir, "meta_state.pt")
        if not os.path.exists(meta_path):
            print(f"No training state found at {meta_path}.")
            return 0, 0
        accelerator.load_state(dir)
        meta = torch.load(meta_path, map_location="cpu")
        epoch = meta.get("epoch", 0)
        global_step = meta.get("global_step", 0)
        print(f"Loaded training state from {meta_path}: epoch={epoch}, global_step={global_step}")
        return epoch, global_step

    def save_training_state(self, accelerator, epoch_id, global_step):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Saving training state...")
        meta_state = {"epoch": epoch_id, "global_step": global_step}
        step_save_path = os.path.join(self.output_path, f"checkpoint-step-{global_step}")
        os.makedirs(step_save_path, exist_ok=True)
        meta_path = os.path.join(step_save_path, "meta_state.pt")
        if accelerator.is_main_process:
            torch.save(meta_state, meta_path)
        accelerator.save_state(step_save_path)


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    def validate(
        self,
        accelerator,
        global_step,
        args,
        test_dataloader=None,
        output_path=None,
        validate_batch=1,
    ):
        """Validate model by generating sample videos and computing metrics."""
        from diffsynth.utils.data import save_video

        rank = accelerator.process_index
        world_size = accelerator.num_processes
        num_inference_steps = 50

        # Create save path
        save_path = os.path.join(output_path, f"validation_results_{num_inference_steps}_inference_steps")
        os.makedirs(save_path, exist_ok=True)

        # If using camera pose file for validation
        if getattr(args, "validate_camera_pose_file", None):
            from PIL import Image
            # Use a default input image if available from validation dataloader
            input_image = None
            input_image_tensor = None
            input_video_tensor = None
            if test_dataloader is not None:
                for batch in test_dataloader:
                    if isinstance(batch, dict) and "input_image" in batch:
                        input_image_tensor = batch["input_image"]
                        if isinstance(input_image_tensor, torch.Tensor):
                            input_image_tensor = input_image_tensor[0] if input_image_tensor.dim() > 3 else input_image_tensor
                        print(f"[DEBUG] Found input_image, shape: {input_image_tensor.shape}")
                        break
                    # If no input_image, try to extract from first frame of video
                    if input_image_tensor is None and isinstance(batch, dict) and "video" in batch:
                        input_video = batch["video"]
                        print(f"[DEBUG] Found video, type: {type(input_video)}")
                        if isinstance(input_video, list) and len(input_video) > 0:
                            # LoadVideo returns list of PIL Images
                            input_image = input_video[0]  # Take first frame as PIL Image
                            print(f"[DEBUG] Using first frame from video list, type: {type(input_image)}")
                        elif isinstance(input_video, torch.Tensor):
                            print(f"[DEBUG] Video tensor shape: {input_video.shape}")
                            if input_video.dim() == 5:
                                # (B, C, T, H, W) -> take first frame
                                input_image_tensor = input_video[0, :, 0, :, :]
                            elif input_video.dim() == 4:
                                if input_video.shape[1] == 3:
                                    input_image_tensor = input_video[0]
                                else:
                                    input_image_tensor = input_video[0, 0]
                            if input_image_tensor is not None:
                                print(f"[DEBUG] Extracted input_image tensor, shape: {input_image_tensor.shape}")

            if input_image_tensor is None and input_image is None:
                # Create a dummy image if no input available
                input_image_tensor = torch.zeros(3, args.height, args.width)

            # Convert tensor to PIL Image for pipeline
            if input_image_tensor is not None and isinstance(input_image_tensor, torch.Tensor):
                # CHW -> HWC -> PIL Image
                input_image_tensor = input_image_tensor.cpu().float()
                if input_image_tensor.dim() == 3:
                    input_image_tensor = input_image_tensor.permute(1, 2, 0)
                input_image_tensor = (input_image_tensor * 255).clamp(0, 255).byte()
                input_image = Image.fromarray(input_image_tensor.cpu().numpy(), mode="RGB")

            # Generate video with camera pose file
            rgb_video_save_num = rank
            with torch.no_grad():
                videos = self.pipe(
                    prompt="",
                    negative_prompt="",
                    input_image=input_image,
                    height=args.height,
                    width=args.width,
                    num_frames=getattr(args, "validate_num_frames", 61),
                    num_inference_steps=num_inference_steps,
                    tiled=True,
                    seed=0,
                    camera_control_pose_file=args.validate_camera_pose_file,
                )

            if videos is not None:
                predict_save_path = os.path.join(
                    save_path,
                    f"video_pose_{rgb_video_save_num}_step_{global_step}.mp4",
                )
                print(f"Rank {rank} saving video to {predict_save_path}")
                save_video(videos, predict_save_path, fps=10, quality=5)
            return

        # Original validation logic using dataloader
        assert test_dataloader is not None, "Test dataloader must be provided for validation."

        for num_inference_step in [num_inference_steps]:
            rgb_video_save_num = rank * args.batch_size

            for idx, batch in enumerate(test_dataloader):
                if idx >= validate_batch:
                    break
                if idx != 0:
                    rgb_video_save_num += (world_size - 1) * args.batch_size

                with torch.no_grad():
                    input_data = self.get_pipeline_inputs(batch)
                    input_data = self.transfer_data_to_device(input_data, self.pipe.device, self.pipe.torch_dtype)

                    # Generate video using pipeline
                    videos = self.pipe(
                        prompt=input_data.get("prompt", [""] * input_data.get("batch_size", 1)),
                        negative_prompt=[""] * len(input_data.get("prompt", [""])),
                        input_image=input_data.get("input_image"),
                        input_video=input_data.get("input_video"),
                        height=input_data.get("height"),
                        width=input_data.get("width"),
                        num_frames=input_data.get("num_frames"),
                        num_inference_steps=num_inference_step,
                        tiled=True,
                        seed=0,
                    )

                    # Save generated video
                    if videos is not None:
                        predict_save_path = os.path.join(
                            save_path,
                            f"video_{rgb_video_save_num}_step_{global_step}.mp4",
                        )
                        print(f"Rank {rank} saving video to {predict_save_path}")
                        save_video(videos, predict_save_path, fps=10, quality=5)
                        rgb_video_save_num += 1


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    # Validation parameters
    parser.add_argument("--validate_step", type=int, default=500, help="Validate and save checkpoint every N steps.")
    parser.add_argument("--log_step", type=int, default=10, help="Log loss every N steps.")
    parser.add_argument("--init_validate", action="store_true", default=False, help="Whether to validate before starting training.")
    parser.add_argument("--validate_batch", type=int, default=1, help="Number of batches to validate.")
    parser.add_argument("--validation_dataset_metadata_path", type=str, default=None, help="Path to validation dataset metadata.")
    parser.add_argument("--validate_camera_pose_file", type=str, default=None, help="Path to camera pose file for validation (REALESTATE10K format).")
    parser.add_argument("--validate_num_frames", type=int, default=61, help="Number of frames to generate during validation.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    # Create training dataset
    train_dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )

    # Create validation dataset (if validation metadata is provided)
    val_dataloader = None
    if args.validation_dataset_metadata_path is not None:
        val_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.validation_dataset_metadata_path,
            repeat=1,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            }
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers
        )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    # Custom training loop with validation
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # Load training state if exists
    start_epoch, global_step = model_logger.load_training_state(accelerator, dir=None)
    accelerator.print(f"Starting training from epoch {start_epoch}, step {global_step}")

    # Initial validation
    if args.init_validate and (val_dataloader is not None or args.validate_camera_pose_file is not None):
        accelerator.print("Running initial validation...")
        model.pipe.dit.eval()
        model.validate(
            accelerator=accelerator,
            global_step=global_step,
            args=args,
            test_dataloader=val_dataloader,
            output_path=model_logger.output_path,
            validate_batch=args.validate_batch,
        )
        model.pipe.scheduler.set_timesteps(1000, training=True)
        model.pipe.dit.train()
        accelerator.wait_for_everyone()

    accumulate_loss = 0.0
    acm_cnt = 0

    for epoch_id in range(start_epoch, args.num_epochs):
        train_dataloader.set_epoch(epoch_id)
        for data in tqdm(
            train_dataloader,
            desc=f"Epoch {epoch_id + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        ):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if train_dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.trainable_modules(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    global_step += 1
                    accumulate_loss += loss.detach().clone()
                    acm_cnt += 1

                    # Log loss
                    if global_step % args.log_step == 0 and acm_cnt > 0:
                        avg_loss = (accumulate_loss / acm_cnt).item()
                        accumulate_loss = 0.0
                        acm_cnt = 0
                        accelerator.print(f"Step {global_step}, Loss: {avg_loss:.6f}")

                    # Validate and save checkpoint
                    if global_step % args.validate_step == 0:
                        # Save checkpoint
                        model_logger.save_training_state(
                            accelerator=accelerator,
                            epoch_id=epoch_id,
                            global_step=global_step,
                        )
                        accelerator.print(f"Checkpoint saved at step {global_step}")

                        # Run validation
                        if val_dataloader is not None or args.validate_camera_pose_file is not None:
                            model.pipe.dit.eval()
                            model.validate(
                                accelerator=accelerator,
                                global_step=global_step,
                                args=args,
                                test_dataloader=val_dataloader,
                                output_path=model_logger.output_path,
                                validate_batch=args.validate_batch,
                            )
                            model.pipe.scheduler.set_timesteps(1000, training=True)
                            model.pipe.dit.train()
                        accelerator.wait_for_everyone()

    accelerator.print("Training completed!")
    model_logger.save_training_state(accelerator, epoch_id, global_step)
