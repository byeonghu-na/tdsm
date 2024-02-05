# This code is borrowed from Karras et al. (EDM, https://github.com/NVlabs/edm/blob/main/Dockerfile).
# The original code is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

FROM nvcr.io/nvidia/pytorch:22.10-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0

WORKDIR /workspace

RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]
