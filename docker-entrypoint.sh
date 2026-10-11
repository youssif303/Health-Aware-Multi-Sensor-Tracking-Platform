#!/bin/sh
set -eu

# Render mounts persistent disks over these directories at runtime. Fix their
# top-level ownership before dropping privileges to the application user.
mkdir -p /app/data /mlflow
chown app:app /app/data /mlflow
exec gosu app "$@"
