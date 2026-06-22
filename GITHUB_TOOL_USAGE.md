# GitHub Push Tool Usage

This workspace uses a project-local SSH wrapper for GitHub access. Do not put GitHub credentials in global user locations on this shared server.

## Security Boundary

All GitHub-related private state for this project must stay under:

```text
/home/ldj/wxc/.private/
```

The private directory is ignored by git:

```text
.private/
```

Do not copy the private key into:

```text
~/.ssh/
~/.gitconfig
/etc/
```

Do not commit `.private/`, `.codex/`, datasets, checkpoints, archives, or server-local configs.

## Files

```text
/home/ldj/wxc/.private/github/<github-private-key>      private SSH key, mode 600
/home/ldj/wxc/.private/github/<github-public-key>.pub   public SSH key
/home/ldj/wxc/.private/ssh/known_hosts.github            GitHub host keys
/home/ldj/wxc/.private/ssh/git-ssh-dispatch              git SSH dispatcher
```

The dispatcher sends GitHub traffic through the project-local GitHub key. Non-GitHub traffic falls back to the existing NAS SSH wrapper when available.

## Current Remote

```text
github git@github.com:wang-xinchun/2d-gaussian-image-processing.git
nas    ssh://nas_git@100.101.227.38/volume2/Basic_1T/git/sdnu.git
```

## Use From This Worktree

The main worktree already has repository-local git config:

```bash
git config --local core.sshCommand /home/ldj/wxc/.private/ssh/git-ssh-dispatch
```

Verify GitHub connectivity:

```bash
GIT_SSH_COMMAND=/home/ldj/wxc/.private/ssh/git-ssh-dispatch \
  ssh -T git@github.com
```

List remote heads:

```bash
GIT_SSH_COMMAND=/home/ldj/wxc/.private/ssh/git-ssh-dispatch \
  git ls-remote --heads git@github.com:wang-xinchun/2d-gaussian-image-processing.git
```

## Use From Another Worktree

In another worktree or clone on this server, configure only that repository:

```bash
git config --local core.sshCommand /home/ldj/wxc/.private/ssh/git-ssh-dispatch
git remote add github git@github.com:wang-xinchun/2d-gaussian-image-processing.git
```

If the remote already exists:

```bash
git remote set-url github git@github.com:wang-xinchun/2d-gaussian-image-processing.git
```

Never use `git config --global` for this project.

## Before Push

Before pushing project code, audit what will be included:

```bash
git status --short
git diff --cached --stat
git ls-files --cached
```

Reject accidental large or private files before commit:

```bash
git check-ignore -v .private .codex
find . -type f -size +50M -not -path './.git/*'
```

Do not blindly run:

```bash
git add .
```

Prefer adding scoped paths after checking `.gitignore`.

## Push Test Status

GitHub push access has been verified. A temporary test branch was pushed and then deleted. The remote now has `main` as `HEAD`.

## If Access Must Be Revoked

Remove the SSH key from GitHub:

```text
GitHub -> Settings -> SSH and GPG keys
```

Then delete the project-local key:

```bash
rm -f /home/ldj/wxc/.private/github/<github-private-key>
rm -f /home/ldj/wxc/.private/github/<github-public-key>.pub
```
