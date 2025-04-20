## Introducing viPiper: Advancing TTS Training and Optimization

Built upon the strong foundation of the Piper TTS project, **viPiper** represents our contribution towards a more flexible and controllable TTS development experience. We've focused on enhancing the training pipeline and introducing novel features for model optimization, shared with the community in the spirit of open collaboration.

**What's New in Version 1.0:**

*   **Enhanced Developer Experience:** A cleaner, more organized codebase.
*   **Precision Finetuning:** Expanded parameters and monitoring tools provide deeper insights and control over the training process.
*   **Efficient Checkpointing:** Save precisely the checkpoints you need with `max_epoch_keeps`.
*   **Audible Progress:** Regularly generated audio samples (`sample_steps`) let you hear the model evolve during training.

**Distinctive Features:**

*   **Adaptive Model Pruning:** viPiper empowers users to prune models to specific dimensions post-training. This is particularly effective for reducing the footprint of HIGH quality models without drastic quality loss.
*   **Smart Initialization:** Whether training from scratch or finetuning, our weight initialization techniques aim for faster convergence and better results.

**Future Contributions:**

Stay tuned! We are actively training a new model using viPiper's capabilities and look forward to sharing it soon.

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
