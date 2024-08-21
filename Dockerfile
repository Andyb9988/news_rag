# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim 

# Prevents Python from writing pyc files.
WORKDIR /youtube_to_gpt

COPY . ./
# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
     python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
# Expose the port that the application listens on.
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# Run the application.
# CMD streamlit run multipage_app/Homepage.py
ENTRYPOINT ["streamlit", "run", "multipage_app/Homepage.py", "--server.port=8080", "--server.address=0.0.0.0"]
