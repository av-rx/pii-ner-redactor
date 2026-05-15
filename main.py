import os
import re
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

LOCAL_CACHE = os.path.join(os.getcwd(), "hf_cache")
os.makedirs(LOCAL_CACHE, exist_ok=True)
os.environ["HF_HOME"] = LOCAL_CACHE

EMAIL_PATTERN = r'\b[\w\.-]+@[\w\.-]+\.\w+\b'
PHONE_PATTERN = r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b'
SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
# loose match: number + street type + optional apt; broad enough for international variants
ADDRESS_PATTERN = (
    r'\b(?:\d{1,5}\s+)?'
    r'(?:[A-Za-z0-9\.\-\s]{1,50})'
    r'(?:\s(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Wy|Square|Sq|Place|Pl|Terrace|Ter))'
    r'(?:[\s,]*(?:Apt|Unit|Suite|#)\s?\w+)?'
)
CREDIT_CARD_PATTERN = r'\b(?:\d[ -]*?){13,16}\b'
PASSPORT_PATTERN = r'\b[A-PR-WY][1-9]\d\s?\d{4}[1-9]\b'

ner_pipeline = None
loader_queue = queue.Queue()


def load_ner_model_in_background(model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    global ner_pipeline
    try:
        loader_queue.put(("status", "Importing transformers..."))
        # Import inside thread to avoid blocking main thread at startup
        from transformers import pipeline  # local import inside thread

        loader_queue.put(("status", f"Loading model: {model_name}"))
        ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple", device=-1)  # -1 = CPU
        loader_queue.put(("done", "Model loaded."))
    except Exception as exc:
        ner_pipeline = None
        loader_queue.put(("error", f"Model failed to load: {exc}"))


def redact_text(text: str) -> str:
    redacted_text = text
    offset = 0

    # ML-based redaction if available
    if ner_pipeline:
        try:
            entities = ner_pipeline(text)
            # entities are usually in order of appearance; we still handle offsets
            for ent in entities:
                start = ent.get("start")
                end = ent.get("end")
                if start is None or end is None:
                    continue
                start += offset
                end += offset
                label = ent.get("entity_group", "ENTITY")
                redaction = f"[REDACTED_{label}]"
                redacted_text = redacted_text[:start] + redaction + redacted_text[end:]
                offset += len(redaction) - (end - start)
        except Exception:
            # If ML fails unexpectedly, proceed with regex-only
            pass

    # Regex runs after ML, catching structured PII the model misses
    patterns = [
        (EMAIL_PATTERN, "[REDACTED_EMAIL]"),
        (PHONE_PATTERN, "[REDACTED_PHONE]"),
        (ADDRESS_PATTERN, "[REDACTED_ADDRESS]"),
        (SSN_PATTERN, "[REDACTED_SSN]"),
        (CREDIT_CARD_PATTERN, "[REDACTED_CREDIT_CARD]"),
        (PASSPORT_PATTERN, "[REDACTED_PASSPORT]")
    ]
    for pat, repl in patterns:
        redacted_text = re.sub(pat, repl, redacted_text, flags=re.IGNORECASE)

    return redacted_text


class PiiRedactorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PII NER Redactor")
        self.geometry("1100x650")
        self.minsize(900, 500)

        self._create_widgets()
        self._start_model_loader()  # start background loading immediately
        self.after(200, self._poll_loader_queue)  # poll for model load updates

    def _create_widgets(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Paned window for side-by-side text panels
        self.paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left: original
        frame_left = ttk.Labelframe(self.paned, text="Original Text")
        frame_left.pack(fill=tk.BOTH, expand=True)
        self.text_original = tk.Text(frame_left, wrap=tk.WORD)
        self.text_original.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scroll_l = ttk.Scrollbar(frame_left, orient=tk.VERTICAL, command=self.text_original.yview)
        self.text_original.configure(yscrollcommand=scroll_l.set)
        scroll_l.pack(side=tk.RIGHT, fill=tk.Y)
        self.paned.add(frame_left, weight=1)

        # Right: redacted
        frame_right = ttk.Labelframe(self.paned, text="Redacted Text")
        frame_right.pack(fill=tk.BOTH, expand=True)
        self.text_redacted = tk.Text(frame_right, wrap=tk.WORD, fg="darkred")
        self.text_redacted.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scroll_r = ttk.Scrollbar(frame_right, orient=tk.VERTICAL, command=self.text_redacted.yview)
        self.text_redacted.configure(yscrollcommand=scroll_r.set)
        scroll_r.pack(side=tk.RIGHT, fill=tk.Y)
        self.paned.add(frame_right, weight=1)

        # Bottom controls
        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=8)

        self.load_btn = ttk.Button(controls, text="Load .TXT", command=self._on_load_file)
        self.load_btn.pack(side=tk.LEFT, padx=6)

        self.redact_btn = ttk.Button(controls, text="Redact", command=self._on_redact)
        self.redact_btn.pack(side=tk.LEFT, padx=6)

        self.save_btn = ttk.Button(controls, text="Save Redacted", command=self._on_save_file)
        self.save_btn.pack(side=tk.LEFT, padx=6)

        # Model status indicator
        self.status_var = tk.StringVar(value="Model: Loading...")
        self.status_label = ttk.Label(controls, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=8)

    def _start_model_loader(self):
        threading.Thread(target=load_ner_model_in_background, daemon=True).start()
        self._show_loading_dialog()

    def _show_loading_dialog(self):
        self.loading_win = tk.Toplevel(self)
        self.loading_win.title("Loading NER Model")
        self.loading_win.geometry("360x130")
        self.loading_win.transient(self)
        # non-modal so the main GUI stays responsive while the model downloads
        self.loading_win.attributes("-topmost", True)

        lbl = ttk.Label(
            self.loading_win,
            text="Model is loading in background.\nYou can use the app; ML will activate when ready.",
            wraplength=320,
        )
        lbl.pack(padx=12, pady=(12, 6))

        self.loading_status_var = tk.StringVar(value="Starting...")
        status_lbl = ttk.Label(self.loading_win, textvariable=self.loading_status_var, foreground="blue")
        status_lbl.pack(padx=12, pady=6)

        ttk.Button(self.loading_win, text="Hide", command=self.loading_win.withdraw).pack(pady=(0, 8))

    def _poll_loader_queue(self):
        try:
            while True:
                kind, msg = loader_queue.get_nowait()
                if kind == "status":
                    self.loading_status_var.set(msg)
                    self.status_var.set(f"Model: {msg}")
                elif kind == "done":
                    self.loading_status_var.set("Model ready.")
                    self.status_var.set("Model: Ready")
                    try:
                        self.loading_win.after(600, self.loading_win.destroy)
                    except Exception:
                        pass
                elif kind == "error":
                    self.loading_status_var.set("Model failed to load.")
                    self.status_var.set("Model: Failed")
                    messagebox.showerror("Model Error", msg)
                    try:
                        self.loading_win.after(600, self.loading_win.destroy)
                    except Exception:
                        pass
        except queue.Empty:
            pass
        # keep polling
        self.after(200, self._poll_loader_queue)

    def _on_load_file(self):
        path = filedialog.askopenfilename(title="Open Text File", filetypes=[("Text Files", "*.txt")])
        if not path:
            return
        if not path.lower().endswith(".txt"):
            messagebox.showerror("Invalid File", "Only .txt files are supported.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
            self.text_original.delete(1.0, tk.END)
            self.text_original.insert(tk.END, data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def _on_redact(self):
        original = self.text_original.get(1.0, tk.END).rstrip()
        if not original:
            messagebox.showwarning("No Input", "Please enter or load text to redact.")
            return
        redacted = redact_text(original)
        self.text_redacted.delete(1.0, tk.END)
        self.text_redacted.insert(tk.END, redacted)

    def _on_save_file(self):
        redacted = self.text_redacted.get(1.0, tk.END).rstrip()
        if not redacted:
            messagebox.showwarning("Nothing to Save", "Redacted text is empty. Redact something first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Save Redacted Text",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(redacted)
            messagebox.showinfo("Saved", f"Redacted text saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file:\n{e}")


if __name__ == "__main__":
    app = PiiRedactorGUI()
    app.mainloop()
