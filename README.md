## Introducing viPiper: Advancing TTS Training and Optimization

Built upon the strong foundation of the Piper TTS project (https://github.com/rhasspy/piper), **viPiper** represents our contribution towards a more flexible and controllable TTS development experience. We've focused on enhancing the training pipeline and introducing novel features for model optimization, shared with the community in the spirit of open collaboration.

**What's New in Version 1.0:**

**Distinctive Features:**

*   **Adaptive Model Pruning:** viPiper empowers users to prune models to specific dimensions post-training. This is particularly effective for reducing the footprint of HIGH quality models without drastic quality loss.
*   **Smart Initialization:** Whether training from scratch or finetuning, our weight initialization techniques aim for faster convergence and better results.
*   **Enhanced Developer Experience:** A cleaner, more organized codebase.
*   **Precision Finetuning:** Expanded parameters and monitoring tools provide deeper insights and control over the training process.
*   **Efficient Checkpointing:** Save precisely the checkpoints you need with `max_epoch_keeps`.
*   **Audible Progress:** Regularly generated audio samples (`sample_steps`) let you hear the model evolve during training.
*   **Streaming**: simple streaming FastAPI and HTML/Javascript and NodJS server-side
*   A clean, fully checked **docker**
```
docker run --rm -it --gpus '"device=0,1,2,3"' --shm-size 64G -v ./data:/data thusinh1969/piper_prune_v1:latest bash
```
*   A comprehensive **starting script** (run inside docker) can be as sophisticated as for training **from scratch** for medium-spec model on 4xRTX3090 24G:
```
WORLD_SIZE=4 python3 -m piper_train --dataset-dir /data/piper/steve_combined_multi_extra_char/to_train \
--default_root_dir  /data/piper/steve_combined_multi_extra_char/to_train/outputs \
--weights_save_path /data/piper/steve_combined_multi_extra_char/to_train/outputs/weights \
--quality medium \
--keep-layers 6 \
--pruning-factor 0.0 \
--precision 32 \
--amp_backend native \
--accelerator 'gpu' --devices 4 --strategy ddp --seed 42 --enable_checkpointing true --batch-size 32 --max-phoneme-ids 400 --accumulate_grad_batches 2 \
--max_epochs 500 --validation-split 0.005 --num-test-examples 100 --check_val_every_n_epoch 1 --log_every_n_steps 10 --logger true \
--gradient_clip_val 1.0 --gradient_clip_algorithm 'norm' \
--auto_lr_find true --learning_rate 2e-4 --weight_decay 0.02 --warmup_ratio 0.05 --cosine_scheduler true --from_scratch \
--sample_steps 5000 \
--checkpoint-epochs 1 \
--max_epoch_keeps 50 \
--num_workers 8 \
--enable_progress_bar false
```

**Streaming:**
```
# pip install fastapi uvicorn piper-tts langchain underthesea vinorm
# Edit f5tts-fastapi-server.py and change model checkpoint, reference audio and reference text (as many as you like and name them well), then simply call:

python f5tts-fastapi-server.py <-- check port inside this file
```

From browser call the host/port and you are up and running.

**Future Contributions:**
*  Stay tuned! We are actively training a new Vietnamese model using viPiper's capabilities and look forward to sharing it soon.
*  Contact us for contribution or commercial implementation at nguyen@hatto.com

**License:**

viPiper is available under the **MIT License**.

**Show Your Support & Cite Our Work:**

Your feedback and acknowledgement motivate us! If viPiper aids your research or project, please consider giving our [GitHub repository](https://github.com/EraX-AI/viF5TTS) a star ⭐.

To cite viPiper in your publications:

```bibtex
@misc{viPiper_EraX_2025,
  author       = {Steve Nguyen Anh Nguyen},
  title        = {viPiper: Enhanced TTS Training, Pruning, and Vietnamese Voice Synthesis},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
