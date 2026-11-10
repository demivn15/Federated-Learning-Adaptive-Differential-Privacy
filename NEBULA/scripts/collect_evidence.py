"""Wait for the three submitted local NEBULA checks and save their evidence.

This collects existing scenarios. It does not submit or restart experiments.
"""
import argparse
import getpass
import http.cookiejar
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

def collect(output, timeout, titles, username):
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    password = os.environ.get("NEBULA_DEMO_PASSWORD") or getpass.getpass("Local NEBULA password: ")
    body = urllib.parse.urlencode({"user": username, "password": password}).encode()
    for attempt in range(12):
        try:
            with opener.open("http://127.0.0.1:6060/platform/login", body, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError("Local NEBULA sign-in failed")
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 11:
                raise
            print(json.dumps({"waiting_for_local_sign_in": attempt + 1}), flush=True)
            time.sleep(5)
    def get(path):
        with opener.open("http://127.0.0.1:6060" + path, timeout=10) as response:
            return json.load(response)
    deadline, previous, finished = time.monotonic() + timeout, {}, {}
    output.mkdir(parents=True, exist_ok=True)
    while time.monotonic() < deadline:
        try:
            scenarios = get("/platform/api/dashboard")
            if not isinstance(scenarios, list):
                raise RuntimeError("The scenario list is unavailable")
            for scenario in scenarios:
                if scenario["title"] not in titles or scenario["title"] in finished:
                    continue
                name = scenario["name"]
                payload = get(f"/platform/api/dashboard/{urllib.parse.quote(name)}/research")
                signature = (scenario["status"], len(payload["privacy"]), len(payload["evaluations"]))
                if previous.get(name) != signature:
                    print(json.dumps({"title": scenario["title"], "name": name,
                        "status": signature[0], "privacy_rows": signature[1],
                        "evaluations": signature[2], "errors": payload["errors"]}), flush=True)
                    previous[name] = signature
                if payload["errors"] or scenario["status"] == "failed":
                    (output / f"{name}_failed_evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    raise RuntimeError("A private scenario failed. Its evidence was preserved.")
                if scenario["status"] not in {"completed", "finished"} or payload["final_test"] is None:
                    continue
                if payload["dp"]["enabled"] and not payload["accounting_complete"]:
                    continue
                if payload["protocol"] != "project2-public-evaluation-v3":
                    raise RuntimeError("The submitted scenario did not apply protocol v3")
                if len(payload["evaluations"]) != payload["rounds"]:
                    raise RuntimeError("The planned global evaluations are incomplete")
                (output / f"{name}_evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                finished[scenario["title"]] = payload
            if titles == finished.keys():
                keys = {json.dumps(p["comparison_key"], sort_keys=True) for p in finished.values()}
                if len(keys) != 1:
                    raise RuntimeError("Completed runs have different base configurations")
                summary = [{"title": p["scenario"]["title"], "scenario": p["scenario"]["name"],
                    "mode": p["dp"]["mode"] if p["dp"]["enabled"] else "baseline",
                    "execution": p["execution"], "final_test": p["final_test"],
                    "epsilon_max": p["epsilon_max"],
                    "round_seconds_sum": sum(r["round_seconds"] for r in p["evaluations"]),
                    "client1_clips": [r["clip"] for r in p["privacy"] if r["participant"] == 1]}
                    for p in finished.values()]
                (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps({"completed": len(finished), "summary": summary}), flush=True)
                return
        except (urllib.error.URLError, TimeoutError) as error:
            print(json.dumps({"waiting_for_local_service": str(error)}), flush=True)
        time.sleep(15)
    raise TimeoutError("The evidence collection deadline elapsed. Existing experiment data was preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--title", action="append", required=True, help="Scenario title to collect, repeat for each run")
    parser.add_argument("--user", default="admin")
    args = parser.parse_args()
    collect(args.output, args.timeout, set(args.title), args.user)
