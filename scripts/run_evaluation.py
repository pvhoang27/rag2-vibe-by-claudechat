"""
scripts/run_evaluation.py
--------------------------
CLI script to run RAGAS evaluation with pre-built sample QA pairs.
Usage: python scripts/run_evaluation.py
"""

import asyncio
import argparse
import contextlib
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ── Sample QA pairs (question + ground truth) ──────────────────────────────────
EVAL_SAMPLES = [
    {
        "question": "AI là gì và được phát minh vào năm nào?",
        "ground_truth": "Trí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính tập trung vào xây dựng hệ thống thông minh. AI được khai sinh vào năm 1956 tại Hội nghị Dartmouth."
    },
    {
        "question": "Machine Learning có những loại nào?",
        "ground_truth": "Machine Learning có 3 loại chính: Supervised Learning (học có giám sát), Unsupervised Learning (học không giám sát), và Reinforcement Learning (học tăng cường)."
    },
    {
        "question": "RAG là gì và tại sao cần dùng RAG thay vì LLM thuần túy?",
        "ground_truth": "RAG (Retrieval-Augmented Generation) kết hợp khả năng sinh văn bản của LLM với truy xuất thông tin từ cơ sở dữ liệu. Cần RAG vì LLM thuần túy bị giới hạn bởi knowledge cutoff, có thể hallucination và không có kiến thức về tài liệu nội bộ."
    },
    {
        "question": "RAGAS đánh giá RAG bằng những metrics nào?",
        "ground_truth": "RAGAS đánh giá RAG với 4 metrics: Faithfulness (câu trả lời dựa trên context), Answer Relevancy (liên quan câu hỏi), Context Recall (context bao phủ thông tin cần thiết), Context Precision (context chính xác, ít nhiễu)."
    },
    {
        "question": "ChromaDB là gì và ưu điểm của nó?",
        "ground_truth": "ChromaDB là vector database open-source, nhẹ, phù hợp cho prototype. Nó được dùng để lưu trữ embedding vectors trong hệ thống RAG."
    },
    {
        "question": "LangChain dùng để làm gì?",
        "ground_truth": "LangChain là framework xây dựng ứng dụng với LLM, hỗ trợ RAG, Agents, Chains và tích hợp với hầu hết LLM và vector stores."
    },
    {
        "question": "Supervised Learning khác Unsupervised Learning như thế nào?",
        "ground_truth": "Supervised Learning được huấn luyện trên dữ liệu đã gán nhãn để phân loại hoặc dự đoán, còn Unsupervised Learning tự tìm kiếm cấu trúc ẩn trong dữ liệu không có nhãn."
    },
    {
        "question": "Python có những thư viện AI nào phổ biến?",
        "ground_truth": "Các thư viện AI Python phổ biến gồm: NumPy, Pandas, Scikit-learn, PyTorch, TensorFlow/Keras, LangChain, và HuggingFace Transformers."
    },
]

DEFAULT_SAMPLE_COUNT = 3


def _looks_like_ollama_error(status: int, detail: str) -> bool:
    detail_l = (detail or "").lower()
    if status not in {500, 502, 503}:
        return False
    return any(
        marker in detail_l
        for marker in [
            "ollama",
            "11434",
            "connection refused",
            "winerror 10061",
            "/api/embeddings",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against local API.")
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of QA samples to evaluate (default: {DEFAULT_SAMPLE_COUNT}, max: {len(EVAL_SAMPLES)})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="API base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--stall-timeout",
        type=int,
        default=180,
        help="Auto-abort if progress does not change for N seconds (default: 180, set 0 to disable)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast evaluation mode (fewer metrics, better for weak machines).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional output file tag (e.g. --tag weak_pc_test).",
    )
    return parser.parse_args()


def _render_progress_bar(percent: float, width: int = 24) -> str:
    filled = int((max(0.0, min(100.0, percent)) / 100.0) * width)
    return "█" * filled + "░" * (width - filled)


async def _progress_polling(
    base_url: str,
    stop_event: asyncio.Event,
    abort_event: asyncio.Event,
    stall_timeout_sec: int,
    interval: int = 2,
) -> None:
    """Poll /eval/progress and print true server-side percent progress."""
    import httpx

    start = time.perf_counter()
    last_signature = None
    last_change_at = time.perf_counter()
    last_printed_stall_sec = -1
    while not stop_event.is_set():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{base_url}/eval/progress")
                r.raise_for_status()
                payload = r.json()

            percent = float(payload.get("percent", 0.0) or 0.0)
            stage = str(payload.get("stage", "running") or "running")
            is_running = bool(payload.get("is_running", False))
            done = int(payload.get("completed_samples", 0) or 0)
            total = int(payload.get("total_samples", 0) or 0)
            message = str(payload.get("message", "") or "")

            if not is_running and stage == "idle" and percent <= 0.0:
                # Skip noisy pre-run idle states.
                await asyncio.sleep(0.2)
                continue

            signature = (round(percent, 2), stage, done, total, message)
            if signature != last_signature:
                last_change_at = time.perf_counter()
                elapsed = int(time.perf_counter() - start)
                mm, ss = divmod(elapsed, 60)
                bar = _render_progress_bar(percent)
                sample_text = f" {done}/{total}" if total > 0 else ""
                console.print(
                    f"   [cyan]{percent:6.2f}%[/cyan] [{bar}] [dim]{stage}{sample_text} | {mm:02d}:{ss:02d}[/dim]"
                )
                if message:
                    console.print(f"   [dim]{message}[/dim]")
                last_signature = signature

            if stall_timeout_sec > 0 and stage not in {"done", "failed"}:
                stalled_for = int(time.perf_counter() - last_change_at)
                if stalled_for >= stall_timeout_sec and not abort_event.is_set():
                    abort_event.set()
                    sm, ss = divmod(stalled_for, 60)
                    console.print(
                        f"   [red]✗ No progress change for {sm:02d}:{ss:02d}. Auto-aborting request.[/red]"
                    )
                elif stalled_for >= max(10, stall_timeout_sec // 2):
                    if stalled_for != last_printed_stall_sec and stalled_for % 10 == 0:
                        sm, ss = divmod(stalled_for, 60)
                        console.print(
                            f"   [dim]No progress change for {sm:02d}:{ss:02d}...[/dim]"
                        )
                        last_printed_stall_sec = stalled_for
        except Exception:
            # Keep polling even if server is briefly unavailable during reload.
            pass

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def _wait_for_server(base_url: str, attempts: int = 5, timeout_sec: int = 5) -> bool:
    """Retry health checks to survive brief server reload windows."""
    import httpx

    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                r = await client.get(f"{base_url}/health")
                r.raise_for_status()
            return True
        except Exception as exc:
            if attempt == attempts:
                detail = str(exc).strip() or repr(exc)
                console.print(f"   [red]✗ Server not reachable: {detail}[/red]")
                return False
            console.print(
                f"   [dim]Health check attempt {attempt}/{attempts} failed, retrying in 2s...[/dim]"
            )
            await asyncio.sleep(2)
    return False


async def main():
    import httpx

    args = _parse_args()
    requested_samples = max(1, min(args.samples, len(EVAL_SAMPLES)))
    selected_samples = EVAL_SAMPLES[:requested_samples]

    BASE_URL = args.base_url.rstrip("/")
    stall_timeout_sec = max(0, args.stall_timeout)
    mode = "fast" if args.fast else "full"

    console.print(Panel.fit(
        "[bold cyan]RAG Chatbot — RAGAS Evaluation Runner[/bold cyan]\n"
        f"Evaluating {len(selected_samples)}/{len(EVAL_SAMPLES)} QA pairs",
        border_style="cyan"
    ))

    # Check server health
    console.print("\n[yellow]1. Checking server health…[/yellow]")
    if await _wait_for_server(BASE_URL, attempts=5, timeout_sec=5):
        console.print("   [green]✓ Server is running[/green]")
    else:
        console.print("   [dim]Please start the server with: uv run uvicorn app.main:app --reload[/dim]")
        console.print(f"   [dim]Current BASE_URL: {BASE_URL}[/dim]")
        sys.exit(1)

    # Check collection has documents
    console.print("\n[yellow]2. Checking vector collection…[/yellow]")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE_URL}/chat/collection")
        info = r.json()
        doc_count = info.get("document_count", 0)
        console.print(f"   [green]✓ Collection '{info['name']}' has {doc_count} chunks[/green]")
        if doc_count == 0:
            console.print("   [red]✗ No documents found! Please ingest data first.[/red]")
            console.print("   [dim]Run: python scripts/generate_sample_data.py[/dim]")
            console.print("   [dim]Then: POST /ingest/directory via API[/dim]")
            sys.exit(1)

    # Run evaluation
    console.print(f"\n[yellow]3. Running RAGAS evaluation on {len(selected_samples)} samples ({mode} mode)…[/yellow]")
    console.print("   [dim]This may take several minutes depending on your machine…[/dim]\n")

    payload = {"samples": selected_samples}
    stop_event = asyncio.Event()
    abort_event = asyncio.Event()
    progress_task = asyncio.create_task(
        _progress_polling(
            BASE_URL,
            stop_event,
            abort_event,
            stall_timeout_sec=stall_timeout_sec,
            interval=2,
        )
    )

    report = None
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            request_task = asyncio.create_task(
                client.post(
                    f"{BASE_URL}/eval/run",
                    params={
                        "mode": mode,
                        "tag": (args.tag or "").strip() or None,
                    },
                    json=payload,
                )
            )
            while not request_task.done():
                if abort_event.is_set():
                    request_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await request_task
                    console.print(
                        "   [yellow]Tip:[/yellow] reduce samples further (e.g. [bold]--samples 1[/bold]) "
                        "or disable auto-abort with [bold]--stall-timeout 0[/bold]."
                    )
                    sys.exit(1)
                await asyncio.sleep(0.5)

            response = await request_task
            response.raise_for_status()
            report = response.json()
    except httpx.ReadTimeout:
        console.print("   [red]✗ Request timeout after 10 minutes (600s).[/red]")
        console.print("   [dim]Tip: reduce sample count or optimize /eval/run on server side.[/dim]")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = ""
        try:
            body = exc.response.json()
            detail = body.get("detail", "") if isinstance(body, dict) else str(body)
        except Exception:
            detail = exc.response.text

        console.print(f"   [red]✗ Server returned HTTP {status} for /eval/run.[/red]")
        if detail:
            console.print(f"   [red]Detail:[/red] {detail}")
        if _looks_like_ollama_error(status, detail):
            console.print("   [yellow]Ollama appears to be offline or unreachable.[/yellow]")
            console.print("   [dim]Run in another terminal:[/dim] [bold]ollama serve[/bold]")
            console.print(
                "   [dim]Then ensure required models exist:[/dim] "
                "[bold]ollama pull llama3.2:3b[/bold] and [bold]ollama pull nomic-embed-text[/bold]"
            )
            console.print("   [dim]Quick check:[/dim] [bold]ollama list[/bold]")
        console.print("   [dim]Check API logs to see the stack trace of the failing metric/model call.[/dim]")
        sys.exit(1)
    finally:
        stop_event.set()
        await progress_task

    # Display results table
    table = Table(
        title="\n📊 RAGAS Evaluation Report",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", width=22)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Grade", justify="center", width=10)
    table.add_column("Description", style="dim")

    for metric in report["metrics"]:
        score = metric["score"]
        if score >= 0.8:
            grade, color = "Excellent", "green"
        elif score >= 0.6:
            grade, color = "Good", "yellow"
        elif score >= 0.4:
            grade, color = "Fair", "orange3"
        else:
            grade, color = "Poor", "red"

        table.add_row(
            metric["name"],
            f"[{color}]{score:.4f}[/{color}]",
            f"[{color}]{grade}[/{color}]",
            metric["description"],
        )

    table.add_section()
    overall = report["overall_score"]
    o_color = "green" if overall >= 0.7 else "yellow" if overall >= 0.5 else "red"
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold {o_color}]{overall:.4f}[/bold {o_color}]",
        "",
        f"Model: {report['model']} | Samples: {report['sample_count']}",
    )

    console.print(table)

    if report.get("output_path"):
        console.print(f"\n[dim]Full report saved to: {report['output_path']}[/dim]")

    console.print("\n[bold green]✅ Evaluation complete![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
