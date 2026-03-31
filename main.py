import os

# Suppress TensorFlow warnings/info logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras import layers, models

# Import scikit-learn for evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

# Base model parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 15  # Adjust training epochs here


class CoughAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Disease CNN Analyzer (Audio & Spectrogram)")
        self.root.geometry("850x800")

        # Variables to store file paths
        self.dataset_path = tk.StringVar()
        self.input_file_path = tk.StringVar()
        self.model_path = tk.StringVar(value="lung_disease_cnn_model.h5")

        self.class_names = []
        self.prediction_results = []
        self.log_queue = queue.Queue()

        self.create_widgets()
        self.check_queue()  # Start checking logs immediately

    def create_widgets(self):
        # Apply style
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 10))

        # Create Tabs
        tab_control = ttk.Notebook(self.root)
        self.tab_train = ttk.Frame(tab_control)
        self.tab_predict = ttk.Frame(tab_control)

        tab_control.add(self.tab_train, text='⚙️ 1. Train Model')
        tab_control.add(self.tab_predict, text='🔍 2. Predict & Visualize')
        tab_control.pack(expand=1, fill='both')

        self.setup_train_tab()
        self.setup_predict_tab()

    # ========================== TAB 1: TRAIN ==========================
    def setup_train_tab(self):
        frame = ttk.LabelFrame(self.tab_train, text="Training Setup", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Select Dataset Folder (containing subfolders):").grid(row=0, column=0, sticky="w",
                                                                                     pady=5)
        ttk.Entry(frame, textvariable=self.dataset_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, pady=5)

        ttk.Label(frame, text="Save Model As (.h5):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.model_path, width=50).grid(row=1, column=1, padx=5, pady=5)

        self.btn_train = ttk.Button(frame, text="🚀 Start Training (Train 80% : Val 20%)",
                                    command=self.start_training_thread)
        self.btn_train.grid(row=2, column=0, columnspan=3, pady=10)

        # Log text box
        self.log_text = tk.Text(self.tab_train, height=25, width=95, bg='#1e1e1e', fg='#00ff00', font=('Courier', 10))
        self.log_text.pack(padx=10, pady=10)
        self.log_text.insert(tk.END, "Ready for training...\n")
        self.log_text.config(state='disabled')

    # ========================== TAB 2: PREDICT ==========================
    def setup_predict_tab(self):
        frame = ttk.LabelFrame(self.tab_predict, text="Upload & Predict", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Audio (.wav, .mp3) or Image (.png, .jpg) File:").grid(row=0, column=0, sticky="w",
                                                                                     pady=5)
        ttk.Entry(frame, textvariable=self.input_file_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse File", command=self.browse_input_file).grid(row=0, column=2, pady=5)

        self.btn_predict = ttk.Button(frame, text="📊 Predict Data", command=self.predict_data)
        self.btn_predict.grid(row=1, column=0, columnspan=3, pady=10)

        # Matplotlib canvas area
        self.fig, self.ax = plt.subplots(figsize=(8, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_predict)
        self.canvas.get_tk_widget().pack(padx=10, pady=5)

        # Save to CSV button
        self.btn_save_csv = ttk.Button(self.tab_predict, text="💾 Save Results to CSV", command=self.save_to_csv,
                                       state='disabled')
        self.btn_save_csv.pack(pady=10)

    # ========================== UTILITIES ==========================
    def browse_dataset(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path.set(folder)

    def browse_input_file(self):
        file = filedialog.askopenfilename(
            filetypes=[("All Supported Files", "*.wav *.mp3 *.png *.jpg *.jpeg"),
                       ("Audio Files", "*.wav *.mp3"),
                       ("Image Files", "*.png *.jpg *.jpeg")]
        )
        if file:
            self.input_file_path.set(file)

    def log(self, message):
        self.log_queue.put(message)

    def check_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state='disabled')
        self.root.after(100, self.check_queue)

    # ========================== TRAINING LOGIC ==========================
    def start_training_thread(self):
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset folder first.")
            return
        self.btn_train.config(state='disabled')
        self.log("\n--- Starting Data Preparation ---")
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        try:
            data_dir = self.dataset_path.get()

            # Load dataset and split 80/20 automatically
            self.log("Loading data and splitting Train 80% / Validation 20%...")
            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                shuffle=False  # Disable shuffle to ensure correct labels matching during evaluation
            )

            self.class_names = train_ds.class_names
            num_classes = len(self.class_names)
            self.log(f"Found {num_classes} classes:")
            for i, name in enumerate(self.class_names):
                self.log(f" - {name}")

            # Save class names for prediction phase
            np.save("class_names.npy", self.class_names)

            # Build CNN Model Structure
            self.log("\nBuilding CNN Model structure...")
            model = models.Sequential([
                layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),  # Prevent Overfitting
                layers.Dense(num_classes, activation='softmax')  # Softmax for probability percentage
            ])

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            self.log(f"Starting training for {EPOCHS} epochs...")

            class CustomLogCallback(tf.keras.callbacks.Callback):
                def __init__(self, logger):
                    self.logger = logger

                def on_epoch_end(self, epoch, logs=None):
                    self.logger(
                        f"Epoch {epoch + 1}/{EPOCHS} -> Loss: {logs['loss']:.4f} | Acc: {logs['accuracy']:.4f} | Val_Loss: {logs['val_loss']:.4f} | Val_Acc: {logs['val_accuracy']:.4f}")

            # Start Training
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=[CustomLogCallback(self.log)],
                verbose=0
            )

            # ================= EVALUATION METRICS =================
            self.log("\n--- Calculating Evaluation Metrics ---")

            y_true = []
            y_pred_probs = []

            # Extract predictions and true labels from the Validation set
            for images, labels in val_ds:
                preds = model.predict(images, verbose=0)
                y_pred_probs.extend(preds)
                y_true.extend(labels.numpy())

            y_true = np.array(y_true)
            y_pred_probs = np.array(y_pred_probs)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)

            # Calculate metrics ('macro' average is used for multi-class classification)
            acc = accuracy_score(y_true, y_pred_classes)
            prec = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred_classes)

            # Calculate AUC-ROC
            try:
                # 'ovr' = One-vs-Rest for Multi-class
                auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
            except ValueError:
                # Triggers if the validation set doesn't contain all classes
                auc = float('nan')

            self.log("Evaluation Results (Validation Set):")
            self.log(f" 🔹 Accuracy   : {acc:.4f} ({(acc * 100):.2f}%)")
            self.log(f" 🔹 Precision  : {prec:.4f}")
            self.log(f" 🔹 Recall     : {rec:.4f}")
            self.log(f" 🔹 F1-Score   : {f1:.4f}")
            self.log(f" 🔹 MCC        : {mcc:.4f}")

            if not np.isnan(auc):
                self.log(f" 🔹 AUC-ROC    : {auc:.4f}")
            else:
                self.log(" 🔹 AUC-ROC    : Cannot be calculated (missing classes in validation set)")

            # Save the Model
            save_path = self.model_path.get()
            model.save(save_path)
            self.log(f"\n✅ Training Complete! Model saved at: {save_path}")
            messagebox.showinfo("Success", "Training, evaluation, and model saving completed successfully!")

        except Exception as e:
            self.log(f"❌ Error occurred: {str(e)}")
            messagebox.showerror("Error", f"Training Error: {str(e)}")
        finally:
            self.btn_train.config(state='normal')

    # ========================== PREDICTION LOGIC ==========================
    def audio_to_spectrogram(self, audio_file):
        # Convert audio to spectrogram image without borders or axes
        temp_img_path = "temp_spectrogram_input.png"
        y, sr = librosa.load(audio_file, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3))
        plt.axis('off')
        librosa.display.specshow(S_dB, sr=sr, cmap='viridis')
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return temp_img_path

    def predict_data(self):
        file_path = self.input_file_path.get()
        model_file = self.model_path.get()

        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid file (audio or image).")
            return

        if not os.path.exists(model_file):
            messagebox.showerror("Error",
                                 "Model file not found. Please train the model first or specify the correct file name.")
            return

        try:
            self.btn_predict.config(text="⏳ Analyzing...", state='disabled')
            self.root.update()

            # 1. Load class names
            if os.path.exists("class_names.npy"):
                self.class_names = np.load("class_names.npy").tolist()
            else:
                messagebox.showerror("Error", "class_names.npy not found (Please train the model again).")
                return

            # 2. Check file extension and prepare image
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.wav', '.mp3']:
                img_path = self.audio_to_spectrogram(file_path)
            elif ext in ['.png', '.jpg', '.jpeg']:
                img_path = file_path
            else:
                messagebox.showerror("Error", "Only .wav, .mp3, .png, .jpg, .jpeg files are supported.")
                return

            # 3. Load model and convert image to array
            model = tf.keras.models.load_model(model_file)
            img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            # 4. Predict results
            predictions = model.predict(img_array)
            scores = predictions[0] * 100  # Convert to percentage

            # Create a list of dictionaries for plotting and CSV export
            self.prediction_results = [{"Disease_Class": self.class_names[i], "Probability(%)": round(scores[i], 2)} for
                                       i in range(len(self.class_names))]

            # Sort results from highest to lowest probability
            self.prediction_results = sorted(self.prediction_results, key=lambda x: x['Probability(%)'], reverse=False)

            # Draw chart
            self.visualize_results(self.prediction_results)

            self.btn_save_csv.config(state='normal')

            # Remove temporary spectrogram file
            if ext in ['.wav', '.mp3'] and os.path.exists("temp_spectrogram_input.png"):
                os.remove("temp_spectrogram_input.png")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred: {str(e)}")
        finally:
            self.btn_predict.config(text="📊 Predict Data", state='normal')

    def visualize_results(self, results):
        self.ax.clear()

        # Extract data for plotting
        classes = [item['Disease_Class'] for item in results]
        scores = [item['Probability(%)'] for item in results]

        # Clean up class numbers (e.g., "1. COVID-19" -> "COVID-19") for better display
        clean_classes = [c.split('. ')[-1] if '. ' in c else c for c in classes]

        # Define colors (highest percentage gets red, others blue)
        colors = ['lightcoral' if i == len(scores) - 1 else 'skyblue' for i in range(len(scores))]

        bars = self.ax.barh(clean_classes, scores, color=colors)
        self.ax.set_xlabel('Probability (%)')
        self.ax.set_title('CNN Model Prediction Results')
        self.ax.set_xlim(0, 100)

        # Add percentage text to bars
        for bar in bars:
            width = bar.get_width()
            self.ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center', fontweight='bold')

        self.fig.tight_layout()
        self.canvas.draw()

    def save_to_csv(self):
        if not self.prediction_results:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="prediction_results.csv"
        )

        if file_path:
            # Sort descending before saving to CSV
            sorted_for_csv = sorted(self.prediction_results, key=lambda x: x['Probability(%)'], reverse=True)
            df = pd.DataFrame(sorted_for_csv)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Success", f"CSV file saved successfully!\nLocation: {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CoughAnalyzerApp(root)
    root.mainloop()