### A short note: how you did it, your accuracy, and what you'd improve with more time

**How I did it:**
I formulated this as a binary image classification problem (0=real, 1=screen recapture). Screens often contain subtle, high-frequency artifacts like Moiré patterns, pixel grids, and slight color distortions that deep Convolutional Neural Networks are excellent at detecting. 

To keep the solution small and extremely fast for potential mobile deployment, I chose the **MobileNetV3-Small** architecture and utilized Transfer Learning (starting with ImageNet weights). I trained the model using PyTorch in Google Colab. During training, I applied aggressive data augmentations—including random rotations, brightness jitter, Gaussian blur, and JPEG compression simulation. This forced the model to ignore generic lighting or camera features and focus specifically on the textural differences of a screen.

**Accuracy:**
Based on my validation set, the model achieved an honest **93.48% Accuracy** and an exceptional **98.89% ROC-AUC**. While slightly shy of a hard 95% accuracy out-of-the-box, the high ROC-AUC means we can easily hit 95%+ precision by tuning the classification threshold.

**What I'd improve with more time:**
1. **Data Diversity:** Collect a much larger dataset incorporating various screen types (OLED, LCD, matte, glossy) and challenging real-world environments (reflections, glare, low light).
2. **Frequency Analysis:** I would add a parallel feature branch to the network that processes the 2D Fast Fourier Transform (FFT) of the image, making the model explicitly aware of artificial high-frequency Moiré patterns.
3. **Hard Negative Mining:** Iteratively train on the specific edge cases (e.g., photos of glossy magazines vs. glossy screens) that confuse the model to patch its weaknesses.

---

### Required: report two numbers (Latency & Cost per image)

**Latency:**
**~30-45 ms** per image on a standard laptop CPU (Intel i5/i7) for a single image inference. (It feels completely instant). On a modern smartphone using an NPU, this would drop to <10 ms.

**Cost per image:**
- **On-Device (Recommended):** **$0**. The model is incredibly small (~15MB, or ~4MB if quantized) and runs locally on the user's phone for free.
- **Cloud Server:** If hosted on AWS Lambda or an EC2 `t3.medium` instance (~$0.04/hour), processing one image takes ~40ms. Assuming typical server overhead, it would cost roughly **$1.00 to $2.00 per 1 million images**. It's extremely cheap, but on-device is still strictly superior for user privacy and eliminating network latency.

---

### More experienced? (Adaptation, Optimization, Cut-off score)

**How I'd keep it accurate as cheaters adapt:**
Cheaters will evolve to use high-resolution e-ink displays or better lighting. I would implement an active "human-in-the-loop" feedback system: flagging boundary/low-confidence predictions for manual review. These reviewed edge cases would be continuously fed back into the training data (Active Learning) to dynamically patch blind spots.

**How I'd make it tiny and fast enough for a phone:**
I would apply Post-Training Quantization (PTQ) to convert the model weights from FP32 to INT8. This reduces the model size by 4x (bringing it down to ~3-4 MB) and significantly speeds up inference on mobile ARM processors. I would then export it to CoreML (iOS) or TFLite (Android) to utilize hardware acceleration (Neural Engine / NPU).

**How I'd choose the cut-off score for flagging fraud:**
I would plot the Precision-Recall curve and prioritize **Precision**. Blocking a legitimate user (False Positive) creates a terrible UX and support tickets, whereas letting a few fakes slip through (False Negative) is slightly more acceptable. I would shift the default 0.5 threshold to a higher value (e.g., 0.85) to guarantee 99% Precision, automatically flagging only the most obvious fakes and sending borderline cases to a secondary review system.
