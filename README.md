# XLabs - Custom Sign Language Detection AI

Welcome to your custom Sign Language Detection project, developed by XLabs. This powerful AI application uses state-of-the-art computer vision to learn and recognize hand gestures in real-time through a webcam.

This guide will walk you through setting up and running the project, with a special focus on how to collect high-quality data to ensure the AI performs with the best possible accuracy.

## Getting Started

Follow these steps to set up the project on your machine.

### Prerequisites

You must have **Python 3.11** installed.

### Installation

1.  Open the **Command Prompt (CMD)** and navigate to the project folder (`xlabs-sign-language-ai`).
2.  **Create a Virtual Environment**: This isolates the project's dependencies from your system. Run the following command:
    ```dos
    py -3.11 -m venv venv
    ```
3.  **Activate the Environment**: You must do this every time you run the project.
    ```dos
    venv\Scripts\activate
    ```
    Your command prompt line should now start with `(venv)`.
4.  **Install Required Libraries**:
    ```dos
    pip install -r requirements.txt
    ```

## How to Run the Project

The process is broken down into three simple steps. Please follow them in order.

### Step 1: The Most Important Step - Collecting Your Data 📸

The accuracy of this AI is **100% dependent on the quality of the data you provide**. To get excellent results, it's crucial to collect clear and consistent images for each sign you want the AI to learn.

To start collecting data, run this command:

```dos
python src/collect_data.py
```

A webcam window will open. Follow these best practices for optimal results:

#### Key Guidelines for Data Collection:

💡 **Good Lighting is Essential**:
- **Do**: Ensure your hand is brightly and evenly lit. Sit facing a window or a bright lamp.
- **Don't**: Sit in a dark room or with a bright light source (like a window) behind you. This creates shadows and makes your hand difficult to see.

🖼️ **Use a Simple, Uncluttered Background**:
- **Do**: Use a plain, single-color wall as your background. This helps the AI focus only on your hand.
- **Don't**: Use a "busy" background with many objects, patterns, or movement.

👋 **Frame Your Hand Correctly**:
- **Do**: Keep your entire hand gesture clearly visible in the center of the frame.
- **Don't**: Let your fingers or parts of your hand get cut off by the edge of the screen.

🎲 **Introduce Slight Variations (Pro-Tip for High Accuracy)**:
- While recording images for a single sign, slowly and slightly move your hand. Change its angle and position a little. This teaches the AI to recognize the sign from different perspectives, making it much more reliable.

🔢 **Aim for Quantity**:
- Collect at least **150-200 images** for each sign you want to teach the AI.

To use the collector: Press a key (e.g., `H` for "Hello"), the system will pause, and then it will start saving images. Repeat for every sign you want to add. Press `q` to quit.

---

### Step 2: Training the AI Model 🧠

Once you are satisfied with your collected data, you need to train the AI model.

To start training, run this command:

```dos
python src/train_model.py
```

This process will take a few minutes. The script will analyze all the images you collected and create a trained model file (`sign_model.h5`) in the `models` folder.

---

### Step 3: Running the Real-Time Detection! ▶️

This is the final step where you can see your custom AI in action.

To run the application, use the following command:

```dos
python src/detect_sign.py
```

A webcam window will open. Make the hand signs you trained the AI on, and it will display its prediction on the screen.

To run with optional voice output, use this command instead:

```dos
python src/detect_sign.py --tts
```

To close the application at any time, simply press the `q` key.

## Customization and Tips

-   **Adding New Signs**: To teach the AI a new sign, simply repeat **Step 1** (collect data for a new key) and then **Step 2** (retrain the model). The new sign will be automatically included.
-   **Improving Accuracy**: If you find the AI is not accurate enough for a specific sign, the best solution is to go back to **Step 1** and collect more high-quality, varied images for that sign, then retrain the model.

---


Thank you for choosing XLabs. We are confident this tool will meet your needs. Should you have any questions, please don't hesitate to reach out at:
contact.xlabsprojects@gmail.com .

