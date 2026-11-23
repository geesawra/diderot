# syntax=docker/dockerfile:1.29@sha256:572cebbc2b15e6f84c077742952674889cb63cd5e9b80abdffcbe587b6f19183

# The build runs on the builder's own platform and cross-compiles for the
# target (CGO is off), so an arm64 image builds without emulation on an
# amd64 builder and the other way round.
FROM --platform=$BUILDPLATFORM docker.io/library/golang:1.27.4-bookworm@sha256:4fa7c968ba93f1635670522dbb01a2c78150443e1385746924566079f36570ea AS build
WORKDIR /src
# Modules download in their own layer, verified against go.sum, so a source
# change doesn't refetch them.
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY . .
ARG VERSION=dev
ARG TARGETOS TARGETARCH
# The build cache persists across builds on the same builder, so a source
# change recompiles only what it touched.
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -trimpath \
    -ldflags "-s -w -X main.Version=${VERSION}" \
    -o /out/diderot ./cmd/diderot
# The licences of everything compiled into the binary, which its own
# licence and theirs require shipping with it. MPL-2.0 modules come with
# their source. segmentio/asm is MIT-0, which asks for no notice and which
# go-licenses cannot classify.
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    go run github.com/google/go-licenses/v2@v2.0.1 save ./cmd/diderot \
    --save_path=/out/licenses --ignore github.com/bluesky-social/diderot --ignore github.com/segmentio/asm

# The atlas CLI's community edition, Apache-2.0 like the rest of the image.
# The default atlas build is under Ariga's commercial terms, which do not
# allow redistributing it.
FROM docker.io/arigaio/atlas:1.4.1-community-alpine@sha256:16c7215ff8ea5a1c356437f99856b7ccf244959f1f4373fa8702639f573b13ef AS atlas

FROM gcr.io/distroless/static-debian12:nonroot@sha256:afa5c872c891853ca7fcf1f12c3edb23f7eeef36189728842dd51042ff57f7ab
ARG VERSION=dev
LABEL org.opencontainers.image.title="diderot" \
      org.opencontainers.image.description="Diderot: retrieval and Ask over an organization's written knowledge" \
      org.opencontainers.image.source="https://github.com/bluesky-social/diderot" \
      org.opencontainers.image.documentation="https://github.com/bluesky-social/diderot/blob/main/docs/production.md" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0" \
      org.opencontainers.image.version="${VERSION}"
COPY --from=build /out/diderot /usr/local/bin/diderot
# atlas is invoked as a subprocess from internal/migrations on
# startup. Migration SQL ships inside the binary via //go:embed;
# atlas just provides the migration engine.
COPY --from=atlas --chmod=0755 /bin/atlas /usr/local/bin/atlas
COPY LICENSE-APACHE LICENSE-MIT LICENSE-DUAL /usr/share/doc/diderot/
COPY --from=build /out/licenses /usr/share/doc/diderot/third-party/
# AWS's RDS CA bundle, for sslmode=verify-full in DIDEROT_DB_URL (required with
# RDS IAM auth, internal/dbconn). Source:
# https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
COPY certs/rds-global-bundle.pem /etc/diderot/rds-global-bundle.pem
USER nonroot:nonroot
# Liveness only: readiness (`probe readyz`) can cost a model call, so it is
# the orchestrator's to schedule, as the Compose quickstart and the
# Kubernetes example do.
HEALTHCHECK --interval=30s --timeout=5s --start-period=2m CMD ["/usr/local/bin/diderot", "probe", "healthz"]
ENTRYPOINT ["/usr/local/bin/diderot"]
CMD ["serve"]
