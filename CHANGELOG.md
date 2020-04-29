# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.2.0]

### Added

- Introduced jupyternotebook executed with Papermill.
- Implemented ChatLM model which is a simple sequence to sequence model using GPT-2.
- Implemented TopPKGenerator to specify both top-p and top-k filtering.

### Removed

- Removed ChatModel. This model will be implemented in the future, but currently it has some bugs. So this model is removed from current version.

## [v0.1.2] - 2020/02/14

### Fixed

- Remove all special tokens from generated text to extract response.

## [v0.1.1]

### Added

- Add license file.

### Changed

- Fix vocab_size and num_albels in the BaseModel training script to adapt Transformers from v2.2.0 to v2.3.0.

## v0.1.0

### Added

- Add scripts for BaseModel and ChatModel.
