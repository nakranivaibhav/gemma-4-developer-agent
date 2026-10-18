# LOCAL sandbox image — NOT the competition image. Built by: docker build -t swebench-sandbox-local:latest sandbox_local
# The competition image + wheel set lack libraries the public repos' own tests import (measured in
# probes/results/gold_verify/). They are put on a SHELF here (/wheels_extra), on top of the untouched competition
# image, because the harness (a) never copies host wheels into a container whose /wheels is already populated and
# (b) refuses to inject pytest plugins (container_setup.skip_base_pkgs lists pytest_httpbin). Nothing is installed at
# build time: sandbox_local/setup.py (the local hook) installs, per repo, only what that repo's tests need.
FROM swebench-sandbox:latest
COPY wheels_extra/ /wheels_extra/
