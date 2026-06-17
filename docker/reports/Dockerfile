# Base Debian (glibc), NO alpine: el paquete `aws-cli` de Alpine venia con un
# expat roto en la imagen (pyexpat/minidom: "XML_SetAllocTrackerActivationThreshold:
# symbol not found"), lo que tumbaba CUALQUIER invocacion de `aws` con un
# traceback de Python -> el `aws s3 sync` del entrypoint nunca corria y
# /reports + /artifacts quedaban vacios aunque S3 tuviera los archivos.
# AWS CLI v2 oficial es glibc, self-contained y soportado: no depende del
# expat/prompt_toolkit del sistema.
FROM nginx:1.27

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         curl unzip ca-certificates dumb-init \
    && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip \
    && apt-get purge -y unzip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# config nginx que sirve /usr/share/nginx/html con autoindex
COPY docker/reports/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/reports/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/entrypoint.sh"]
