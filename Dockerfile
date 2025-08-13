FROM python:3.13

WORKDIR /server
COPY pyproject.toml uv.lock /server/

COPY --link --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv
RUN uv pip install --system .
RUN uv pip install gunicorn --system

COPY . /server/