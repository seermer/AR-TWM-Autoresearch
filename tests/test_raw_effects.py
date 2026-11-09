"""Kernel code changes state only through the recording functions (subproc, fs, archive, vcs), starts threads
only through telemetry.thread/submit, and every `except` re-raises or writes a record (P6). The *_TODO lists
name the sites still to convert, each with its plan task; they empty by the end of Part G (status.py in H1).
PERMANENT holds the exceptions, each with its reason."""
from raw_effects import scan

EFFECTS_TODO = {
}
HOPS_TODO = {
}
EXCEPTS_TODO = {
}
PERMANENT = {
    'archive/commits.py::CommitStore._validate: except KeyError':
        'an unknown clip id: named in the BadClips refusal',
    'cli.py::_run: except PreflightError':
        'a refused start records nothing (decided); the error is printed',
    'cli.py::_run: except RuntimeError':
        'a refused claim records nothing (decided); the error is printed',
    'cli.py::main: except (RunNotFound, PreflightError)':
        'read-only commands record nothing; the error is printed',
    'control.py::Control.alive_pid: except (OSError, ValueError)':
        'no readable pid file: no live loop',
    'fs.py::_place: except FileExistsError':
        'the folder is there',
    'fs.py::_size: except FileNotFoundError':
        'the file is not there',
    'guards.py::_nvidia_smi: subprocess.run':
        'nvidia-smi queries stay plain calls; their readings are recorded where used (E1)',
    'guards.py::smi: subprocess.run':
        'nvidia-smi queries stay plain calls; their readings are recorded where used (E1)',
    'gateway/store.py::_assistant.args: except (TypeError, ValueError)':
        'tool-call arguments that are not JSON: compared as the raw text',
    'hf_upload.py::HfUpload._upload: .write_text':
        'the file list in the upload\'s own temporary folder; it goes up in the last tar',
    'loop.py::Loop._phase: except Exception':
        'Loop._failed records the error and the decision',
    'loop.py::Loop._ensure_root: except KeyError':
        'no root yet: the root.how decision records it',
    'tools/ask.py::finish_reason: except (KeyError, IndexError, TypeError, AttributeError)':
        'no finish reason in the reply, which its llm.try span holds',
    'monitor.py::_descends_from: except psutil.Error':
        'the process ended: not a kernel process on that GPU',
    'monitor.py::process_memory: except (OSError, psutil.Error)':
        'the process ended between the listing and the read: not in the sample',
    'monitor.py::tree_mark: except OSError':
        'the file went between the listing and the stat: not counted',
    'status.py::run_status: except ValueError':
        'no scored node yet: no probabilities; `ar status` records nothing',
    'subproc.py::_shown: except UnicodeDecodeError':
        'not text: recorded by its size',
    'subproc.py::kill_group: except (ProcessLookupError, PermissionError)':
        'the group is gone',
    'subproc.py::proc_running: except (OSError, IndexError)':
        'the process is gone',
    'subproc.py::proc_start_time: except (OSError, IndexError)':
        'the process is gone',
    'subproc.py::terminate: except ProcessLookupError':
        'the process is gone',
    'subproc.py::tmp_link: except FileExistsError':
        'the link exists; checked next',
    'telemetry/tree.py::Trace.refresh: except (json.JSONDecodeError, KeyError, TypeError)':
        'an unreadable line: a --check finding',
    'telemetry/tree.py::writer_alive: except OSError':
        'no boot id here: not checkable',
    'telemetry/writer.py::Writer.__init__: except OSError':
        "the writer's own failure: stderr and a force stop",
    'telemetry/writer.py::Writer._write: except Exception':
        "the writer's own failure: stderr and a force stop",
    'telemetry/writer.py::Writer.store_file: except Exception':
        "the writer's own failure: stderr and a force stop",
    'telemetry/writer.py::Writer.store_file: except FileExistsError':
        'another thread stored the same bytes',
    'telemetry/writer.py::Writer.write: except IndexError':
        'no record left pending',
    'telemetry/writer.py::_boot_id: except OSError':
        'no boot id on this host',
    'telemetry/writer.py::_task: except RuntimeError':
        'no event loop in this thread',
    'tools/captioner.py::submit: .submit':
        'JobQueue.submit, not a pool',
    'tools/gpu_jobs.py::GpuJob.submit: .submit':
        'JobQueue.submit, not a pool',
    'tools/gpu_jobs.py::register_gpu_tools.annotate_camera: .submit':
        'JobQueue.submit, not a pool',
    'tools/gpu_jobs.py::register_gpu_tools.submit: .submit':
        'JobQueue.submit, not a pool',
    'tools/locate_events.py::submit: .submit':
        'JobQueue.submit, not a pool',
    'tools/jobs.py::JobQueue._run_one: except Exception':
        'the job span ends with the exception (_finalize)',
    'tools/jobs.py::JobQueue._run_one: except OSError':
        'the job span ends with the exception (_finalize)',
    'tools/jobs.py::JobQueue._loop: except Exception':
        'the job span ends with the exception (_end -> _finalize)',
    'tools/jobs.py::JobQueue._run_one.report: .touch':
        'the activity clock callback, not a file',
    'tools/vllm_server.py::VllmServer.start: except httpx.HTTPError':
        'the server is not up yet: polled again until it is or its process ends',
    'tools/server.py::ToolKit._call: .touch':
        'the activity clock callback, not a file',
    'tools/server.py::ToolKit.call.start: .run_in_executor':
        'runs the work in copy_context(), which carries the current span',
    'tools/vllm_server.py::_names.name: except psutil.Error':
        'the process is gone',
    'tools/vllm_server.py::gpu_memory_mib: subprocess.run':
        'an nvidia-smi query (E1)',
    'tools/vllm_server.py::gpu_processes.query: subprocess.run':
        'an nvidia-smi query (E1)',
    'tools/vllm_server.py::process_tree: except psutil.Error':
        'the process is gone',
}


def _check(found: set[str], todo: dict[str, str], what: str) -> None:
    new = sorted(found - set(todo) - set(PERMANENT))
    gone = sorted(set(todo) - found)
    assert not new, f"{what} outside the recording functions: {new}"
    assert not gone, f"converted or moved; take them off the list: {gone}"


def test_state_changes_go_through_the_recording_functions():
    _check(scan()[0], EFFECTS_TODO, "raw state changes")


def test_threads_and_pools_carry_the_current_span():
    _check(scan()[1], HOPS_TODO, "raw thread hops")


def test_every_except_reraises_or_records():
    _check(scan()[2], EXCEPTS_TODO, "except clauses that neither re-raise nor record")


def test_every_permanent_exception_is_still_there():
    effects, hops, excepts = scan()
    assert sorted(set(PERMANENT) - effects - hops - excepts) == []


def test_the_scan_finds_each_kind_of_site():
    import ast
    from raw_effects import _Scan
    code = '''
import os, shutil, subprocess, tempfile, threading
def f(p, conn, pool, loop):
    os.remove(p); shutil.rmtree(p); subprocess.run(["x"]); p.write_text("x"); p.replace(p); "a".replace("a", "b")
    open(p, "w"); open(p); p.open("ab"); tempfile.NamedTemporaryFile(delete=False)
    conn.execute("INSERT INTO t VALUES (1)"); conn.execute("SELECT 1"); fs.write_text(p, "x")
    threading.Thread(target=f); pool.submit(f); loop.run_in_executor(None, f)
    try:
        pass
    except OSError:
        pass
    except ValueError:
        telemetry.error(None, "ignored")
    except KeyError:
        raise
'''
    s = _Scan("x.py")
    s.visit(ast.parse(code))
    assert sorted(x.split(": ", 1)[1] for x in s.effects) == sorted(
        ["os.remove", "shutil.rmtree", "subprocess.run", ".write_text", ".replace", "open(write)", ".open(write)",
         "tempfile.NamedTemporaryFile(delete=False)", "execute(SQL write)"])
    assert sorted(x.split(": ", 1)[1] for x in s.hops) == [".run_in_executor", ".submit", "threading.Thread"]
    assert [x.split(": ", 1)[1] for x in s.silent] == ["except OSError"]
