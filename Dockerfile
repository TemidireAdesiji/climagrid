FROM python:3.12-slim

LABEL maintainer="Temidire Adesiji <temidireadesiji@gmail.com>" \
      description="climagrid — climate data, grid-ready" \
      license="Apache-2.0"

WORKDIR /app

# Copy package manifest first so the install layer is cached when only src/ changes
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --root-user-action=ignore .

# Run as non-root
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["climagrid"]
CMD ["--help"]
