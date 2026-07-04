import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from pier.environments.agent_setup import (
    EGRESS_PROXY_PORT,
    EGRESS_PROXY_SERVICE,
    new_proxy_token,
    proxy_environment,
    proxy_policy_env,
)
from pier.environments.docker.docker import DockerEnvironment


def _config_value(config, name):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _as_dict(config):
    if isinstance(config, dict):
        return config
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "dict"):
        return config.dict()
    return {}


def _mcp_server_hosts(task_env_config) -> list[str]:
    servers = _config_value(task_env_config, "mcp_servers") or []
    if not servers:
        config_dict = _as_dict(task_env_config)
        servers = config_dict.get("mcp_servers") or []

    hosts: list[str] = []
    for server in servers:
        url = _config_value(server, "url")
        if not url:
            url = _as_dict(server).get("url")
        if not url:
            continue
        host = urlparse(str(url)).hostname
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _local_llm_hosts() -> list[str]:
    hosts: list[str] = []
    for name in ("OPENAI_BASE_URL", "QWEN_BASE_URL"):
        url = os.environ.get(name)
        if not url:
            continue
        host = urlparse(url).hostname
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _allowlist_with_local_hosts(allowlist, task_env_config):
    hosts = [*_mcp_server_hosts(task_env_config), *_local_llm_hosts()]
    if not hosts:
        return allowlist

    domains = list(getattr(allowlist, "domains", None) or [])
    for host in hosts:
        if host not in domains:
            domains.append(host)

    if hasattr(allowlist, "model_copy"):
        return allowlist.model_copy(update={"domains": domains})
    if hasattr(allowlist, "copy"):
        return allowlist.copy(update={"domains": domains})
    try:
        allowlist.domains = domains
    except Exception:
        pass
    return allowlist


def _squid_bootstrap_command() -> str:
    return r"""#!/usr/bin/env bash
set -eu

printf '%s' "$ALLOWLIST_DOMAINS" | tr ',' '\n' | sed '/^[[:space:]]*$/d' \
  > /tmp/allowed_domains.txt

htpasswd -bc /tmp/squid.passwd agent "$PROXY_TOKEN"

cat > /tmp/squid.conf <<'EOF'
http_port 0.0.0.0:8080
pid_filename /tmp/squid.pid
coredump_dir /tmp

auth_param basic program /usr/lib/squid/basic_ncsa_auth /tmp/squid.passwd
auth_param basic realm PierPolicyProxy
acl authenticated proxy_auth REQUIRED

acl SSL_ports port 443
acl Safe_ports port 80 443 8080 3004 3005 3006 3007 5000
acl CONNECT method CONNECT
acl allowed_domains dstdomain "/tmp/allowed_domains.txt"

http_access deny !Safe_ports
http_access deny CONNECT !SSL_ports
http_access allow authenticated allowed_domains
http_access deny all

cache deny all
access_log stdio:/tmp/squid_access.log
cache_log /tmp/squid_cache.log
log_mime_hdrs off
shutdown_lifetime 1 seconds
EOF

exec squid -N -f /tmp/squid.conf -d 1
"""


def _write_proxy_compose(path: Path, proxy_dir: Path, allowlist, token: str) -> Path:
    proxy_dir.mkdir(parents=True, exist_ok=True)
    (proxy_dir / "Dockerfile").write_text(
        "\n".join(
            [
                "FROM ubuntu:24.04",
                "RUN apt-get update && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
                "apache2-utils ca-certificates squid && "
                "rm -rf /var/lib/apt/lists/*",
                "COPY start-squid.sh /usr/local/bin/start-squid.sh",
                "RUN chmod +x /usr/local/bin/start-squid.sh",
                'CMD ["bash", "/usr/local/bin/start-squid.sh"]',
                "",
            ]
        )
    )
    (proxy_dir / "start-squid.sh").write_text(_squid_bootstrap_command())
    compose = {
        "services": {
            "main": {
                "networks": ["pier-egress-internal"],
                "depends_on": {
                    EGRESS_PROXY_SERVICE: {"condition": "service_healthy"},
                },
            },
            EGRESS_PROXY_SERVICE: {
                "build": {"context": str(proxy_dir.resolve().absolute())},
                "environment": proxy_policy_env(allowlist, token),
                "healthcheck": {
                    "test": ["CMD-SHELL", "bash -lc '</dev/tcp/127.0.0.1/8080'"],
                    "interval": "1s",
                    "timeout": "1s",
                    "retries": 30,
                },
                "networks": ["pier-egress-internal", "default"],
            },
        },
        "networks": {
            "pier-egress-internal": {"internal": True},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


class DockerLlmProxyEnvironment(DockerEnvironment):
    """Docker environment with Pier filtered egress allowing local LLM port 8080."""

    def _prebuilt_toolchain_image(self) -> str | None:
        install = getattr(self, "agent_install_spec", None)
        metadata = getattr(install, "metadata", None) or {}
        if not metadata.get("qsa_prebuilt_toolchain"):
            return None
        image = str(
            metadata.get("qsa_prebuilt_toolchain_image")
            or os.environ.get("QSA_PREBUILT_TOOLCHAIN_IMAGE")
            or "ai-box-deepswe-tools:latest"
        )
        return image

    def _ensure_prebuilt_toolchain_image(self, image: str) -> None:
        if subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            return
        if os.environ.get("QSA_PREBUILT_TOOLCHAIN_AUTO_BUILD", "1") == "0":
            raise RuntimeError(
                f"Prebuilt DeepSWE toolchain image {image!r} is missing. "
                "Build it with scripts/deepswe-tools.Dockerfile or set "
                "QSA_PREBUILT_TOOLCHAIN=0 for network installs."
            )
        dockerfile = Path(
            os.environ.get(
                "QSA_PREBUILT_TOOLCHAIN_DOCKERFILE",
                "scripts/deepswe-tools.Dockerfile",
            )
        )
        context = Path(os.environ.get("QSA_PREBUILT_TOOLCHAIN_CONTEXT", "."))
        version = image.rsplit(":", 1)[1] if ":" in image else "latest"
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile),
                "--build-arg",
                f"SVERKLO_VERSION={version}",
                "-t",
                image,
                str(context),
            ],
            check=True,
        )

    def _prepare_agent_build_context(self) -> None:
        super()._prepare_agent_build_context()
        image = self._prebuilt_toolchain_image()
        if not image or not self._agent_build_context_dir:
            return
        self._ensure_prebuilt_toolchain_image(image)
        dockerfile = self._agent_build_context_dir / "Dockerfile"
        text = dockerfile.read_text()
        if f"COPY --from={image} " in text:
            return
        insert = "\n".join(
            [
                f"COPY --from={image} /usr/local/bin/node /usr/local/bin/node",
                f"COPY --from={image} /usr/local/bin/npm /usr/local/bin/npm",
                f"COPY --from={image} /usr/local/bin/npx /usr/local/bin/npx",
                f"COPY --from={image} /usr/local/bin/corepack /usr/local/bin/corepack",
                f"COPY --from={image} /usr/local/bin/uv /usr/local/bin/uv",
                f"COPY --from={image} /usr/local/bin/uvx /usr/local/bin/uvx",
                f"COPY --from={image} /usr/local/bin/pyright /usr/local/bin/pyright",
                f"COPY --from={image} /usr/local/bin/tsc /usr/local/bin/tsc",
                f"COPY --from={image} /usr/local/bin/biome /usr/local/bin/biome",
                f"COPY --from={image} /usr/local/bin/ast-grep /usr/local/bin/ast-grep",
                f"COPY --from={image} /usr/local/bin/ruff /usr/local/bin/ruff",
                f"COPY --from={image} /usr/local/bin/mypy /usr/local/bin/mypy",
                f"COPY --from={image} /usr/local/bin/gopls /usr/local/bin/gopls",
                f"COPY --from={image} /usr/local/bin/goimports /usr/local/bin/goimports",
                f"COPY --from={image} /usr/local/bin/staticcheck /usr/local/bin/staticcheck",
                f"COPY --from={image} /usr/local/bin/serena /usr/local/bin/serena",
                f"COPY --from={image} /usr/bin/shellcheck /usr/bin/shellcheck",
                f"COPY --from={image} /usr/bin/shfmt /usr/bin/shfmt",
                f"COPY --from={image} /usr/local/lib/node_modules /usr/local/lib/node_modules",
                f"COPY --from={image} /usr/local/lib/python3.11 /usr/local/lib/python3.11",
                f"COPY --from={image} /usr/local/src /usr/local/src",
                f"COPY --from={image} /opt/qsa-tools /opt/qsa-tools",
                "RUN "
                + json.dumps(
                    [
                        "/bin/bash",
                        "-c",
                        "printf '%s\\n' '#!/usr/bin/env bash' "
                        "'exec /usr/local/bin/node "
                        "/usr/local/lib/node_modules/npm/bin/npm-cli.js "
                        "\"$@\"' > /usr/local/bin/npm && "
                        "printf '%s\\n' '#!/usr/bin/env bash' "
                        "'exec /usr/local/bin/node "
                        "/usr/local/lib/node_modules/npm/bin/npx-cli.js "
                        "\"$@\"' > /usr/local/bin/npx && "
                        "printf '%s\\n' '#!/usr/bin/env bash' "
                        "'exec /usr/local/bin/node "
                        "/usr/local/lib/node_modules/corepack/dist/corepack.js "
                        "\"$@\"' > /usr/local/bin/corepack && "
                        "printf '%s\\n' '#!/usr/bin/env bash' "
                        "'exec /usr/local/bin/node "
                        "/usr/local/lib/node_modules/sverklo/dist/bin/sverklo.js "
                        "\"$@\"' > /usr/local/bin/sverklo && "
                        "chmod +x /usr/local/bin/npm /usr/local/bin/npx "
                        "/usr/local/bin/corepack /usr/local/bin/sverklo",
                    ]
                ),
            ]
        )
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("USER "):
                lines[index:index] = insert.splitlines()
                break
        else:
            lines.append(insert)
        dockerfile.write_text("\n".join(lines) + "\n")

    def _prepare_egress_proxy_compose(self) -> None:
        allowlist = _allowlist_with_local_hosts(
            self.network_allowlist, self.task_env_config
        )
        if self.task_env_config.allow_internet or not allowlist.domains:
            return
        if self._uses_compose:
            raise ValueError(
                "Filtered inference egress is currently supported only for Dockerfile "
                "or prebuilt-image tasks, not docker-compose tasks."
            )
        token = new_proxy_token()
        self._egress_proxy_env = proxy_environment(
            token, EGRESS_PROXY_SERVICE, EGRESS_PROXY_PORT
        )
        self._egress_proxy_compose_path = _write_proxy_compose(
            path=self.trial_paths.trial_dir / "docker-compose-egress-proxy.json",
            proxy_dir=self.trial_paths.trial_dir / "egress-proxy",
            allowlist=allowlist,
            token=token,
        )
