"""Queue controlled H3 video probes on a dedicated local ComfyUI port.

The server chooses a policy through its import-time environment flags; run a
new server for each policy. Prints one JSON row per warmup or measured video.
"""

import argparse
import copy
import json
import time
import urllib.error
import urllib.request


SHAPES = (
    ("480p_124f", 864, 480, 124),
    ("480p_345f", 864, 480, 345),
    ("768p_124f", 1344, 768, 124),
    ("768p_345f", 1344, 768, 345),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, help="Existing H3 API graph")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--policy", required=True, choices=("A", "B", "C", "D"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=(999, 1101),
                        help="First seed warms each shape; the rest are measured")
    args = parser.parse_args()
    graph = json.load(open(args.workflow))
    types = {node["class_type"]: node for node in graph.values()}
    required = ("UNETLoader", "CLIPLoader", "MiniMaxH3ImageToVideo", "BasicScheduler",
                "RandomNoise", "MiniMaxH3MemoryEfficientSageAttentionPatch", "SaveVideo")
    if any(t not in types for t in required):
        parser.error("workflow is missing a required H3 node")
    if not all(node["inputs"].get("strength_model", 0) == 0
               for node in graph.values() if node["class_type"] == "LoraLoaderModelOnly"):
        parser.error("disable the acceleration LoRAs before measuring")

    url = f"http://127.0.0.1:{args.port}"

    def get(path):
        with urllib.request.urlopen(url + path, timeout=30) as response:
            return json.load(response)

    def post(path, payload):
        request = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    for label, width, height, frames in SHAPES:
        for rep, seed in enumerate(args.seeds):
            current = copy.deepcopy(graph)
            values = {n["class_type"]: n["inputs"] for n in current.values()}
            values["MiniMaxH3ImageToVideo"]["length"] = frames
            values["BasicScheduler"]["steps"] = args.steps
            values["RandomNoise"]["noise_seed"] = seed
            values["SaveVideo"]["filename_prefix"] = f"ablation/{args.policy}_{label}_{seed}"
            # The reference graph switches its width and height using three
            # primitive inputs; changing their values keeps all graph links.
            for node in current.values():
                title = node.get("_meta", {}).get("title")
                if title == "Use 768p":
                    node["inputs"]["value"] = height == 768
                elif title == "480p width":
                    node["inputs"]["value"] = 864
                elif title == "480p height":
                    node["inputs"]["value"] = 480
                elif title == "768p width":
                    node["inputs"]["value"] = 1344
                elif title == "768p height":
                    node["inputs"]["value"] = 768

            begin = time.perf_counter()
            prompt_id = post("/prompt", {"prompt": current})["prompt_id"]
            while True:
                history = get(f"/history/{prompt_id}").get(prompt_id)
                if history is not None:
                    break
                if time.perf_counter() - begin > 3600:
                    raise TimeoutError(f"{label}: prompt {prompt_id} did not complete")
                time.sleep(2)
            elapsed = time.perf_counter() - begin
            completed = history.get("status", {}).get("completed", False)
            output = history.get("outputs", {})
            videos = [file for node in output.values() for file in node.get("videos", ())]
            row = {"policy": args.policy, "shape": label, "width": width,
                   "height": height, "frames": frames, "steps": args.steps,
                   "seed": seed, "warmup": rep == 0, "prompt_id": prompt_id,
                   "wall_s": round(elapsed, 2), "completed": completed, "videos": videos,
                   "messages": history.get("status", {}).get("messages", ())}
            print(json.dumps(row), flush=True)
            if not completed:
                raise RuntimeError(f"{label}: prompt {prompt_id} failed; inspect server log")


if __name__ == "__main__":
    main()
