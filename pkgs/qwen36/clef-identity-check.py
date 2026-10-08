"""Device-free checks for Clef release pins, checkpoint verification and identity."""

import base64
import hashlib
import http.client
import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import clef_release_model
from qwen36_service import clef

QUESTIONS = {"q": {"type": "noul"}}
SHARD_COUNTS = {"clef": 12, "clef-flash": 4}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def raises(function, *args):
    """The ValueError message of a call that must fail."""
    try:
        function(*args)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"{function.__name__} accepted {args!r}")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def sri(hex_digest):
    return "sha256-" + base64.b64encode(bytes.fromhex(hex_digest)).decode("ascii")


RELEASE_CODE = sha256_hex(Path(clef_release_model.__file__).read_bytes())


class FakeEngine:
    """Replaces the TT engine; the HTTP surface under test never opens a device."""

    def answer(self, request):
        if request.get("model") != "Cloudflare/Clef":
            raise clef.RequestError("model must identify Cloudflare/Clef")
        return {
            "model": request["model"],
            "answers": {"q": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 7, "output_tokens": 0},
        }


def call(port, method, path, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if body is None else {"Content-Type": "application/json"}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), json.loads(response.read())
    finally:
        connection.close()


def fixture_pins():
    return {
        "joint_head.safetensors": clef.VARIANTS["clef"].head_sha256,
        "joint_schema_model.py": RELEASE_CODE,
        "tokenizer.json": sha256_hex(b"tokenizer"),
    }


def check_release_manifest():
    """Pins decode from SRI to hex and are refused unless they bind the variant."""
    revision = clef.VARIANTS["clef"].revision
    pins = fixture_pins()
    files = {name: sri(digest) for name, digest in pins.items()}
    tokenizer = files["tokenizer.json"]
    bad_files = [
        {},
        {**files, "../escape": tokenizer},
        {**files, "revision": tokenizer},
        {**files, "tokenizer.json": "sha512-" + tokenizer.removeprefix("sha256-")},
        {**files, "tokenizer.json": "sha256-!!"},
        {**files, "tokenizer.json": sri("ab" * 20)},
        {**files, "joint_head.safetensors": tokenizer},
    ]
    documents = [
        {"clef": {"revision": "0" * 40, "files": files}},
        {"clef-flash": {"revision": revision, "files": files}},
    ] + [{"clef": {"revision": revision, "files": bad}} for bad in bad_files]
    with tempfile.TemporaryDirectory() as name:
        manifest = Path(name) / "clef-release.json"

        def load(document):
            manifest.write_text(json.dumps(document), encoding="utf-8")
            return clef.load_release_manifest(manifest, "clef")

        require(
            load({"clef": {"revision": revision, "files": files}}) == pins,
            "manifest pins were not decoded to hex digests",
        )
        for document in documents:
            raises(load, document)


def check_checkpoint_verification():
    """Every pinned file is hashed; changed, missing and unpinned entries fail."""
    contents = {"model-00001-of-00002.safetensors": b"shard", "config.json": b"{}"}
    digests = {name: sha256_hex(data) for name, data in contents.items()}
    with tempfile.TemporaryDirectory() as name:
        checkpoint = Path(name)
        for file, data in contents.items():
            (checkpoint / file).write_bytes(data)
        for inert in ("README.md", ".gitattributes", "revision"):
            (checkpoint / inert).write_text("not read", encoding="utf-8")
        (checkpoint / ".cache").mkdir()
        clef.verify_checkpoint_files(checkpoint, digests)

        (checkpoint / "config.json").write_bytes(b'{"changed": true}')
        message = raises(clef.verify_checkpoint_files, checkpoint, digests)
        require("config.json" in message, f"changed file not named: {message}")
        (checkpoint / "config.json").write_bytes(contents["config.json"])

        (checkpoint / "added_tokens.json").write_text("{}", encoding="utf-8")
        message = raises(clef.verify_checkpoint_files, checkpoint, digests)
        require("added_tokens.json" in message, f"unpinned file not named: {message}")
        (checkpoint / "added_tokens.json").unlink()

        (checkpoint / "model-00001-of-00002.safetensors").unlink()
        message = raises(clef.verify_checkpoint_files, checkpoint, digests)
        require("model-00001" in message, f"missing file not named: {message}")


def check_tensor_cache_state():
    with tempfile.TemporaryDirectory() as name:
        cache = Path(name)
        require(clef.tensor_cache_state(cache) == "empty-at-startup", "empty cache")
        (cache / "converted.bin").write_bytes(b"tensor")
        require(
            clef.tensor_cache_state(cache) == "preexisting-unattested",
            "a reused tensor cache was reported as converted in this process",
        )


def check_identity():
    """The identity binds verified files and the release code, not caller input."""
    spec = clef.VARIANTS["clef"]
    pins = fixture_pins()
    canonical = json.dumps(pins, sort_keys=True, separators=(",", ":"))
    identity = clef.service_identity("clef", pins, "empty-at-startup", 2048)
    require(
        identity
        == {
            "schema": "clef-tt-identity/1",
            "model": "Cloudflare/Clef",
            "accepted_models": ["clef", "Cloudflare/clef", "Cloudflare/Clef"],
            "variant": "clef",
            "hf_repo": "Cloudflare/clef",
            "hf_revision": spec.revision,
            "checkpoint_files": pins,
            "checkpoint_sha256": sha256_hex(canonical.encode("utf-8")),
            "release_code_sha256": RELEASE_CODE,
            "tt_tensor_cache": "empty-at-startup",
            "max_length": 2048,
            "max_body_bytes": clef.MAX_BODY_BYTES,
            "endpoint": "POST /v1/systemone",
        },
        f"unexpected Clef identity: {identity}",
    )
    raises(
        clef.service_identity,
        "clef",
        {**pins, "joint_schema_model.py": sha256_hex(b"other code")},
        "empty-at-startup",
        2048,
    )
    flash = clef.service_identity("clef-flash", pins, "preexisting-unattested", 1024)
    require(
        (
            flash["model"],
            flash["hf_repo"],
            flash["hf_revision"],
            flash["accepted_models"],
        )
        == (
            "Cloudflare/Clef-Flash",
            "Cloudflare/clef-flash",
            clef.VARIANTS["clef-flash"].revision,
            ["clef-flash", "Cloudflare/clef-flash", "Cloudflare/Clef-Flash"],
        ),
        f"unexpected Clef-Flash identity: {flash}",
    )
    return identity


def check_advertised_names_match_guard(identity):
    for name in identity["accepted_models"]:
        clef._validate_request(
            {"model": name, "state": "s", "questions": QUESTIONS}, "clef"
        )
    for name in ("Cloudflare/Clef-Flash", "clef-flash", "Qwen3.8-27B", None):
        try:
            clef._validate_request(
                {"model": name, "state": "s", "questions": QUESTIONS}, "clef"
            )
        except clef.RequestError:
            continue
        raise AssertionError(f"request guard admitted unadvertised model {name!r}")


def check_http_surface(identity):
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = sha256_hex(canonical.encode("utf-8"))
    require(
        clef.identity_digest(identity) == digest, "identity digest is not canonical"
    )
    expected = {
        "X-Clef-Identity-Sha256": digest,
        "X-Clef-Revision": identity["hf_revision"],
        "X-Clef-Checkpoint-Sha256": identity["checkpoint_sha256"],
        "X-Clef-Tensor-Cache": identity["tt_tensor_cache"],
    }
    handler = clef.make_handler(FakeEngine(), identity)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        valid = {"model": "Cloudflare/Clef", "state": "s", "questions": QUESTIONS}
        cases = [
            ("GET", "/v1/identity", None, 200),
            ("GET", "/v1/models", None, 200),
            ("GET", "/v1/health", None, 404),
            ("POST", "/v1/systemone", valid, 200),
            ("POST", "/v1/systemone", {**valid, "model": "Qwen3.8-27B"}, 400),
            ("POST", "/v1/identity", valid, 404),
        ]
        bodies = {}
        for method, path, payload, status in cases:
            got, headers, body = call(port, method, path, payload)
            require(got == status, f"{method} {path}: status {got}, expected {status}")
            for name, value in expected.items():
                require(
                    headers.get(name) == value,
                    f"{method} {path}: header {name}={headers.get(name)!r}",
                )
            bodies[(method, path, status)] = body
        require(bodies[("GET", "/v1/identity", 200)] == identity, "identity changed")
        models = bodies[("GET", "/v1/models", 200)]
        require(
            [entry["id"] for entry in models["data"]] == ["Cloudflare/Clef"],
            "the models list must name only Clef",
        )
        require(models["data"][0]["identity"] == identity, "models entry lost identity")
        require(
            bodies[("GET", "/v1/health", 404)] == {"error": "unknown endpoint"},
            "an unknown GET route answered",
        )
        require(
            bodies[("POST", "/v1/systemone", 200)]["usage"]
            == {"input_tokens": 7, "output_tokens": 0},
            "the decision body changed",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def check_installed_manifest(path):
    """The package manifest pins whole checkpoints that match clef.py and the code."""
    for variant, spec in clef.VARIANTS.items():
        digests = clef.load_release_manifest(path, variant)
        shards = [name for name in digests if name.startswith("model-")]
        require(
            len(shards) == SHARD_COUNTS[variant],
            f"{variant}: manifest pins {len(shards)} backbone shards",
        )
        for name in ("model.safetensors.index.json", "config.json", "tokenizer.json"):
            require(name in digests, f"{variant}: manifest does not pin {name}")
        require(
            digests["joint_schema_model.py"] == RELEASE_CODE,
            f"{variant}: installed clef_release_model.py is not the pinned code",
        )
        require(digests["joint_head.safetensors"] == spec.head_sha256, variant)


def main():
    check_release_manifest()
    check_checkpoint_verification()
    check_tensor_cache_state()
    identity = check_identity()
    check_advertised_names_match_guard(identity)
    check_http_surface(identity)
    if len(sys.argv) > 1:
        check_installed_manifest(Path(sys.argv[1]))
    print("PASS clef identity and checkpoint pins")


if __name__ == "__main__":
    main()
