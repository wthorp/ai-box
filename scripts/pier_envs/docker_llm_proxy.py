import json
from pathlib import Path

from pier.environments.agent_setup import (
    EGRESS_PROXY_PORT,
    EGRESS_PROXY_SERVICE,
    new_proxy_token,
    proxy_environment,
    proxy_policy_env,
)
from pier.environments.docker.docker import DockerEnvironment


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
acl Safe_ports port 80 443 8080
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

    def _prepare_egress_proxy_compose(self) -> None:
        allowlist = self.network_allowlist
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
