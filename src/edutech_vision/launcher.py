from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Text, Tk, filedialog, messagebox, ttk
import tkinter as tk


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "assets" / "models" / "face_landmarker.task"
YUNET_MODEL_PATH = ROOT_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
SCRFD_MODEL_PATH = ROOT_DIR / "assets" / "models" / "scrfd_2.5g_bnkps.onnx"
BENCHMARK_PATH = ROOT_DIR / "assets" / "benchmarks" / "manifest.json"

BG = "#101820"
PANEL = "#16232d"
CARD = "#1d2c36"
CARD_2 = "#223440"
TEXT = "#edf3f7"
MUTED = "#9fb0bb"
ACCENT = "#5ee0a0"
BLUE = "#58b7ff"
YELLOW = "#f4c542"
RED = "#ff6b6b"


@dataclass(frozen=True)
class Task:
    title: str
    commands: list[list[str]]


class LauncherApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.current_process: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None

        self.camera = StringVar(value="0")
        self.width = StringVar(value="960")
        self.height = StringVar(value="540")
        self.max_faces = StringVar(value="24")
        self.detector = StringVar(value="enhanced")
        self.face_confidence = StringVar(value="auto")
        self.yaw_tolerance = StringVar(value="30")
        self.pitch_tolerance = StringVar(value="20")
        self.demo_duration = StringVar(value="0")
        self.fullscreen = BooleanVar(value=True)
        self.showcase = BooleanVar(value=True)
        self.no_sound = BooleanVar(value=True)
        self.video_path = StringVar(value="")

        self.status_vars: dict[str, StringVar] = {
            "python": StringVar(),
            "model": StringVar(),
            "yunet": StringVar(),
            "scrfd": StringVar(),
            "benchmark": StringVar(),
            "package": StringVar(),
            "results": StringVar(),
        }
        self.status_dots: dict[str, tk.Label] = {}

        self.build()
        self.refresh_status()
        self.root.after(100, self.drain_queue)

    def build(self) -> None:
        self.root.title("EduTech Vision | Central de Controle")
        self.root.geometry("1120x760")
        self.root.minsize(980, 660)
        self.root.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", TEXT)])
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"), foreground=TEXT)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 11), foreground=MUTED)
        style.configure("CardTitle.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 13, "bold"))
        style.configure("CardText.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("TCheckbutton", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground="#0f171e", foreground=TEXT)

        self.header()

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=22, pady=(0, 18))

        left = tk.Frame(main, bg=BG, width=315)
        left.pack(side="left", fill="y", padx=(0, 18))
        left.pack_propagate(False)

        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self.status_panel(left)
        self.quick_actions(left)

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        run_tab = tk.Frame(notebook, bg=BG)
        setup_tab = tk.Frame(notebook, bg=BG)
        report_tab = tk.Frame(notebook, bg=BG)
        docs_tab = tk.Frame(notebook, bg=BG)

        notebook.add(run_tab, text="Executar")
        notebook.add(setup_tab, text="Instalar")
        notebook.add(report_tab, text="Resultados")
        notebook.add(docs_tab, text="Documentos")

        self.run_tab(run_tab)
        self.setup_tab(setup_tab)
        self.report_tab(report_tab)
        self.docs_tab(docs_tab)

        self.console(right)

    def header(self) -> None:
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(fill="x", padx=24, pady=(22, 14))

        title_block = tk.Frame(frame, bg=BG)
        title_block.pack(side="left", fill="x", expand=True)
        ttk.Label(title_block, text="EduTech Vision", style="Title.TLabel").pack(anchor="w")
        self.button(frame, "Atualizar", self.refresh_status, BLUE, width=12).pack(side="right", padx=(8, 0))
        self.button(frame, "Parar", self.stop_process, RED, width=10).pack(side="right", padx=(8, 0))

    def status_panel(self, parent: tk.Frame) -> None:
        card = self.card(parent)
        card.pack(fill="x", pady=(0, 14))
        ttk.Label(card, text="Status", style="CardTitle.TLabel").pack(anchor="w", padx=16, pady=(14, 10))

        for key, label in [
            ("python", "Python"),
            ("model", "Modelo MediaPipe"),
            ("yunet", "YuNet enhanced"),
            ("scrfd", "SCRFD pesquisa"),
            ("benchmark", "Corpus benchmark"),
            ("package", "Pacote"),
            ("results", "Resultados"),
        ]:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", padx=16, pady=6)
            dot = tk.Label(row, text="●", bg=CARD, fg=MUTED, font=("Segoe UI", 13, "bold"))
            dot.pack(side="left")
            self.status_dots[key] = dot
            ttk.Label(row, text=label, style="CardText.TLabel", width=16).pack(side="left", padx=(8, 4))
            ttk.Label(row, textvariable=self.status_vars[key], style="CardText.TLabel").pack(side="left")

    def quick_actions(self, parent: tk.Frame) -> None:
        card = self.card(parent)
        card.pack(fill="x")
        ttk.Label(card, text="Inicio Rapido", style="CardTitle.TLabel").pack(anchor="w", padx=16, pady=(14, 10))
        self.button(card, "Modo Individual", self.run_individual, ACCENT).pack(fill="x", padx=16, pady=5)
        self.button(card, "Modo Plateia", self.run_plateia, BLUE).pack(fill="x", padx=16, pady=5)
        self.button(card, "Modo Demo", self.run_demo, YELLOW, fg="#111820").pack(fill="x", padx=16, pady=5)
        self.button(card, "Diagnostico + camera", self.doctor_camera, CARD_2).pack(fill="x", padx=16, pady=(14, 16))

    def run_tab(self, parent: tk.Frame) -> None:
        grid = tk.Frame(parent, bg=BG)
        grid.pack(fill="both", expand=True, padx=4, pady=12)

        left = self.card(grid)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = self.card(grid)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        ttk.Label(left, text="Camera e Tela", style="CardTitle.TLabel").pack(anchor="w", padx=18, pady=(18, 12))
        self.form_entry(left, "Indice da camera", self.camera)
        self.form_entry(left, "Largura", self.width)
        self.form_entry(left, "Altura", self.height)
        self.check(left, "Painel de apresentacao", self.showcase)
        self.check(left, "Tela cheia", self.fullscreen)
        self.check(left, "Sem som no individual", self.no_sound)
        self.video_picker(left)

        ttk.Label(right, text="Configuracao do Modo", style="CardTitle.TLabel").pack(anchor="w", padx=18, pady=(18, 12))
        self.form_entry(right, "Maximo de faces", self.max_faces)
        self.form_choice(right, "Detector", self.detector, ("enhanced", "mediapipe", "research"))
        self.form_entry(right, "Confianca facial", self.face_confidence)
        self.form_entry(right, "Tolerancia yaw", self.yaw_tolerance)
        self.form_entry(right, "Tolerancia pitch", self.pitch_tolerance)
        self.form_entry(right, "Duracao da demo", self.demo_duration)

        button_row = tk.Frame(right, bg=CARD)
        button_row.pack(fill="x", padx=18, pady=(18, 10))
        self.button(button_row, "Individual", self.run_individual, ACCENT).pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.button(button_row, "Plateia", self.run_plateia, BLUE).pack(side="left", fill="x", expand=True, padx=6)
        self.button(button_row, "Demo", self.run_demo, YELLOW, fg="#111820").pack(side="left", fill="x", expand=True, padx=(6, 0))

    def setup_tab(self, parent: tk.Frame) -> None:
        card = self.card(parent)
        card.pack(fill="both", expand=True, padx=4, pady=12)
        ttk.Label(card, text="Instalacao e Validacao", style="CardTitle.TLabel").pack(anchor="w", padx=18, pady=(18, 8))

        self.big_action(card, "Instalar / reparar", "", self.full_setup)
        self.big_action(card, "Baixar modelos padrao", "", self.download_model)
        self.big_action(card, "Baixar modelo de pesquisa", "", self.download_research_model)
        self.big_action(card, "Baixar corpus de benchmark", "", self.download_benchmarks)
        self.big_action(card, "Diagnostico", "", self.doctor)
        self.big_action(card, "Diagnostico + camera", "", self.doctor_camera)
        self.big_action(card, "Checklist automatica", "", self.preflight)

    def report_tab(self, parent: tk.Frame) -> None:
        card = self.card(parent)
        card.pack(fill="both", expand=True, padx=4, pady=12)
        ttk.Label(card, text="Relatorios e Evidencias", style="CardTitle.TLabel").pack(anchor="w", padx=18, pady=(18, 8))

        self.big_action(card, "Gerar relatorio de sessao", "", self.generate_report)
        self.big_action(card, "Gerar relatorio sintetico", "", self.generate_demo_report)
        self.big_action(card, "Benchmark smoke - Enhanced", "", self.benchmark_smoke)
        self.big_action(card, "Resumo do benchmark", "", lambda: self.open_path(ROOT_DIR / "results" / "benchmark" / "summary.html"))
        self.big_action(card, "Relatorio PDF da sessao", "", lambda: self.open_path(ROOT_DIR / "results" / "session_report.pdf"))
        self.big_action(card, "Pasta de resultados", "", lambda: self.open_path(ROOT_DIR / "results"))

    def docs_tab(self, parent: tk.Frame) -> None:
        card = self.card(parent)
        card.pack(fill="both", expand=True, padx=4, pady=12)
        ttk.Label(card, text="Documentacao", style="CardTitle.TLabel").pack(anchor="w", padx=18, pady=(18, 8))

        self.big_action(card, "README", "", lambda: self.open_path(ROOT_DIR / "README.md"))
        self.big_action(card, "Checklist de apresentacao", "", lambda: self.open_path(ROOT_DIR / "docs" / "CHECKLIST_SOFTWARE_APRESENTACAO.md"))
        self.big_action(card, "Defesa oral", "", lambda: self.open_path(ROOT_DIR / "docs" / "DEFESA_ORAL.md"))
        self.big_action(card, "Relatorio tecnico", "", lambda: self.open_path(ROOT_DIR / "docs" / "RELATORIO_TECNICO.md"))
        self.big_action(card, "Plano de benchmark", "", lambda: self.open_path(ROOT_DIR / "docs" / "PLANO_TESTES_PRIOR_ART.md"))
        self.big_action(card, "Pasta de documentos", "", lambda: self.open_path(ROOT_DIR / "docs"))

    def console(self, parent: tk.Frame) -> None:
        wrap = self.card(parent)
        wrap.pack(fill="both", expand=False, pady=(16, 0))
        header = tk.Frame(wrap, bg=CARD)
        header.pack(fill="x", padx=14, pady=(12, 4))
        ttk.Label(header, text="Log de Comandos", style="CardTitle.TLabel").pack(side="left")
        self.button(header, "Limpar", self.clear_log, CARD_2, width=9).pack(side="right")

        self.log = Text(
            wrap,
            height=10,
            bg="#081117",
            fg="#d9edf7",
            insertbackground=TEXT,
            relief="flat",
            font=("Consolas", 10),
            padx=12,
            pady=10,
        )
        self.log.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        self.write_log("Pronto.\n")

    def card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground="#2b4050")

    def button(self, parent: tk.Widget, text: str, command, bg: str, fg: str = TEXT, width: int | None = None) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=14,
            pady=9,
            width=width or 0,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )

    def form_entry(self, parent: tk.Frame, label: str, var: StringVar) -> None:
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=18, pady=7)
        ttk.Label(row, text=label, style="CardText.TLabel", width=24).pack(side="left")
        entry = tk.Entry(row, textvariable=var, bg="#0f171e", fg=TEXT, insertbackground=TEXT, relief="flat", width=18)
        entry.pack(side="left", fill="x", expand=True, ipady=6)

    def form_choice(self, parent: tk.Frame, label: str, var: StringVar, options: tuple[str, ...]) -> None:
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=18, pady=7)
        ttk.Label(row, text=label, style="CardText.TLabel", width=24).pack(side="left")
        picker = ttk.Combobox(row, textvariable=var, values=options, state="readonly")
        picker.pack(side="left", fill="x", expand=True, ipady=4)

    def check(self, parent: tk.Frame, label: str, var: BooleanVar) -> None:
        item = tk.Checkbutton(
            parent,
            text=label,
            variable=var,
            bg=CARD,
            fg=TEXT,
            selectcolor="#0f171e",
            activebackground=CARD,
            activeforeground=TEXT,
            font=("Segoe UI", 10),
        )
        item.pack(anchor="w", padx=16, pady=4)

    def video_picker(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=18, pady=(10, 16))
        ttk.Label(row, text="Video file", style="CardText.TLabel", width=24).pack(side="left")
        tk.Entry(row, textvariable=self.video_path, bg="#0f171e", fg=TEXT, insertbackground=TEXT, relief="flat").pack(
            side="left", fill="x", expand=True, ipady=6
        )
        self.button(row, "Browse", self.pick_video, CARD_2, width=8).pack(side="left", padx=(8, 0))

    def big_action(self, parent: tk.Frame, title: str, _description: str, command) -> None:
        row = tk.Frame(parent, bg=CARD_2)
        row.pack(fill="x", padx=18, pady=7)
        text = tk.Frame(row, bg=CARD_2)
        text.pack(side="left", fill="x", expand=True, padx=14, pady=11)
        tk.Label(text, text=title, bg=CARD_2, fg=TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.button(row, "Executar", command, ACCENT, fg="#111820", width=9).pack(side="right", padx=12, pady=10)

    def refresh_status(self) -> None:
        py_ok = (3, 10) <= sys.version_info[:2] < (3, 12)
        self.set_status("python", sys.version.split()[0], py_ok)
        model_ok = MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0
        model_text = "ready" if model_ok else "missing"
        self.set_status("model", model_text, model_ok)
        yunet_ok = YUNET_MODEL_PATH.exists() and YUNET_MODEL_PATH.stat().st_size > 0
        self.set_status("yunet", "ready" if yunet_ok else "missing", yunet_ok)
        scrfd_ok = SCRFD_MODEL_PATH.exists() and SCRFD_MODEL_PATH.stat().st_size > 0
        self.set_status("scrfd", "optional ready" if scrfd_ok else "optional", True)
        benchmark_ok = BENCHMARK_PATH.exists()
        self.set_status("benchmark", "ready" if benchmark_ok else "optional", True)
        package_ok = (ROOT_DIR / "src" / "edutech_vision").exists()
        self.set_status("package", "source OK" if package_ok else "missing", package_ok)
        results_ok = (ROOT_DIR / "results").exists()
        self.set_status("results", "folder OK" if results_ok else "missing", results_ok)

    def set_status(self, key: str, value: str, ok: bool) -> None:
        self.status_vars[key].set(value)
        self.status_dots[key].configure(fg=ACCENT if ok else RED)

    def base_args(self, mode: str) -> list[str]:
        confidence = self.face_confidence.get().strip()
        if not confidence or confidence.lower() == "auto":
            confidence = "0.70" if mode == "individual" else "0.60"
        args = [
            "--camera",
            self.camera.get().strip() or "0",
            "--width",
            self.width.get().strip() or "960",
            "--height",
            self.height.get().strip() or "540",
            "--detector",
            self.detector.get().strip() or "enhanced",
            "--face-confidence",
            confidence,
        ]
        video = self.video_path.get().strip()
        if video:
            args.extend(["--video", video])
        if self.showcase.get():
            args.append("--showcase")
        if self.fullscreen.get():
            args.append("--fullscreen")
        return args

    def run_individual(self) -> None:
        args = [sys.executable, "-m", "edutech_vision", "--mode", "individual", *self.base_args("individual")]
        if self.no_sound.get():
            args.append("--no-sound")
        self.run_task(Task("Modo Individual", [*self.detector_setup_commands(), args]))

    def run_plateia(self) -> None:
        args = [
            sys.executable,
            "-m",
            "edutech_vision",
            "--mode",
            "plateia",
            *self.base_args("plateia"),
            "--max-faces",
            self.max_faces.get().strip() or "24",
            "--audience-yaw-tolerance",
            self.yaw_tolerance.get().strip() or "30",
            "--audience-pitch-tolerance",
            self.pitch_tolerance.get().strip() or "20",
        ]
        self.run_task(Task("Modo Plateia", [*self.detector_setup_commands(), args]))

    def run_demo(self) -> None:
        args = [sys.executable, "-m", "edutech_vision", "--mode", "demo"]
        if self.showcase.get():
            args.append("--showcase")
        if self.fullscreen.get():
            args.append("--fullscreen")
        duration = self.demo_duration.get().strip()
        if duration and duration != "0":
            args.extend(["--demo-duration", duration])
        self.run_task(Task("Modo Demo", [args]))

    def full_setup(self) -> None:
        self.run_task(
            Task(
                "Instalar / reparar",
                [
                    [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    [sys.executable, "-m", "pip", "install", "-e", "."],
                    [sys.executable, "scripts/download_models.py"],
                    [sys.executable, "scripts/doctor.py"],
                ],
            )
        )

    def download_model(self) -> None:
        self.run_task(Task("Baixar modelos padrao", [[sys.executable, "scripts/download_models.py"]]))

    def download_research_model(self) -> None:
        self.run_task(Task("Baixar modelo de pesquisa", [[sys.executable, "scripts/download_models.py", "--research"]]))

    def download_benchmarks(self) -> None:
        self.run_task(Task("Baixar corpus de benchmark", [[sys.executable, "scripts/download_benchmarks.py"]]))

    def detector_setup_commands(self) -> list[list[str]]:
        if self.detector.get().strip() == "research" and not SCRFD_MODEL_PATH.exists():
            return [[sys.executable, "scripts/download_models.py", "--research"]]
        return []

    def doctor(self) -> None:
        self.run_task(Task("Doctor", [[sys.executable, "scripts/doctor.py"]]))

    def doctor_camera(self) -> None:
        self.run_task(Task("Doctor + camera", [[sys.executable, "scripts/doctor.py", "--camera-check", "--camera", self.camera.get().strip() or "0"]]))

    def preflight(self) -> None:
        self.run_task(Task("Checklist automatica", [[sys.executable, "scripts/presentation_check.py", "--skip-slow"]]))

    def generate_report(self) -> None:
        self.run_task(Task("Gerar relatorio de sessao", [[sys.executable, "scripts/generate_session_report.py"]]))

    def generate_demo_report(self) -> None:
        self.run_task(Task("Gerar relatorio sintetico", [[sys.executable, "scripts/generate_session_report.py", "--demo-synthetic"]]))

    def benchmark_smoke(self) -> None:
        self.run_task(
            Task(
                "Benchmark smoke - Enhanced",
                [
                    [sys.executable, "scripts/benchmark_vision.py", "--suite", "smoke", "--mode", "individual", "--detector", "enhanced"],
                    [sys.executable, "scripts/benchmark_vision.py", "--suite", "smoke", "--mode", "plateia", "--detector", "enhanced"],
                ],
            )
        )

    def run_task(self, task: Task) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("EduTech Vision", "Uma tarefa ja esta em execucao.")
            return
        self.write_log(f"\n== {task.title} ==\n")
        self.worker = threading.Thread(target=self._run_task_worker, args=(task,), daemon=True)
        self.worker.start()

    def _run_task_worker(self, task: Task) -> None:
        try:
            for command in task.commands:
                self.queue.put(("log", f"$ {format_command(command)}\n"))
                self.current_process = subprocess.Popen(
                    command,
                    cwd=ROOT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert self.current_process.stdout is not None
                for line in self.current_process.stdout:
                    if visible_process_line(line):
                        self.queue.put(("log", line))
                return_code = self.current_process.wait()
                self.current_process = None
                if return_code != 0:
                    self.queue.put(("log", f"[falhou com codigo {return_code}]\n"))
                    break
            else:
                self.queue.put(("log", "[concluido]\n"))
        finally:
            self.current_process = None
            self.queue.put(("status", "refresh"))

    def stop_process(self) -> None:
        process = self.current_process
        if process is None:
            self.write_log("[nenhum processo em execucao]\n")
            return
        process.terminate()
        self.write_log("[parada solicitada]\n")

    def drain_queue(self) -> None:
        try:
            while True:
                kind, value = self.queue.get_nowait()
                if kind == "log":
                    self.write_log(value)
                elif kind == "status":
                    self.refresh_status()
        except queue.Empty:
            pass
        self.root.after(100, self.drain_queue)

    def write_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def clear_log(self) -> None:
        self.log.delete("1.0", "end")

    def pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecionar video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.video_path.set(path)

    def open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showwarning("EduTech Vision", f"Arquivo nao encontrado:\n{path}")
            return
        os.startfile(path)  # type: ignore[attr-defined]


def format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def visible_process_line(line: str) -> bool:
    native_noise = (
        "face_landmarker_graph.cc",
        "Created TensorFlow Lite XNNPACK delegate",
        "inference_feedback_manager.cc",
        "portable_clearcut_uploader.cc",
        "Source Location Trace",
        "wireless/android/play/playlog",
    )
    return not any(marker in line for marker in native_noise)


def main() -> None:
    root = Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
