import argparse
import re
import shutil
import subprocess
import sys


# --- ANSI Color Codes for better output ---
class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_info(message: str):
    """Prints an informational message."""
    print(f"{colors.OKBLUE}INFO: {message}{colors.ENDC}")


def print_success(message: str):
    """Prints a success message."""
    print(f"{colors.OKGREEN}{colors.BOLD}SUCCESS: {message}{colors.ENDC}")


def print_error(message: str):
    """Prints an error message and exits."""
    print(f"{colors.FAIL}{colors.BOLD}ERROR: {message}{colors.ENDC}", file=sys.stderr)
    sys.exit(1)


def print_command(command: str):
    """Prints a command that is about to be run."""
    print(f"{colors.HEADER}EXEC: {command}{colors.ENDC}")


def check_prerequisites():
    """Check if 'uv' and 'git' are installed and if we're in a git repo."""
    print_info("Checking prerequisites...")
    if not shutil.which("uv"):
        print_error(
            "'uv' command not found. Please install it: https://docs.astral.sh/uv/install/"
        )
    if not shutil.which("git"):
        print_error(
            "'git' command not found. Please ensure git is installed and in your PATH."
        )

    try:
        # Check if inside a git repository
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("This script must be run from within a git repository.")
    print_success("Prerequisites met.")


def run_command(
    command: list[str], dry_run: bool = False, capture_output: bool = False
) -> str:
    """
    Executes a shell command, handles errors, and supports a dry-run mode.
    """
    cmd_str = " ".join(command)
    print_command(cmd_str)

    if dry_run:
        return "DRY_RUN_OUTPUT"

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if capture_output:
            return result.stdout.strip()
        # Print stdout for non-capturing commands for user feedback
        if result.stdout:
            print(result.stdout)
        return ""
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with exit code {e.returncode}:\n{e.stderr}")
    except FileNotFoundError:
        print_error(f"Command not found: {command[0]}")
    sys.exit(1)


def parse_new_version(uv_output: str) -> str:
    """
    Parses the output from 'uv version' (e.g., "1.0.0 => 1.0.1") to get the new version.
    """
    match = re.search(r"\s=>\s([\w.-]+(?:a|b)?\d*)$", uv_output)
    if not match:
        print_error(f"Could not parse the new version from 'uv' output: '{uv_output}'")
        sys.exit(1)

    new_version = match.group(1)
    print_info(f"Detected new version: {new_version}")
    return new_version


def main():
    """Main function to parse arguments and run the publishing workflow."""
    parser = argparse.ArgumentParser(
        description="Automate package versioning and git tagging using 'uv' and 'git'.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    version_group = parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument(
        "--bump",
        choices=["patch", "minor", "major"],
        help="Increment the version by 'patch', 'minor', or 'major'.",
    )
    version_group.add_argument(
        "--set-version",
        metavar="VERSION",
        help="Set a specific version number (e.g., '1.2.3').",
    )

    parser.add_argument(
        "--prerelease",
        choices=["alpha", "beta"],
        help="Add a pre-release identifier (e.g., 'alpha', 'beta').",
    )

    # --- Control Flow Arguments ---
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push the new tag to the remote repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without running them.",
    )

    args = parser.parse_args()

    if not args.dry_run:
        check_prerequisites()

    # 1. Construct and run the 'uv version' command
    uv_command = ["uv", "version"]
    if args.set_version:
        uv_command.append(args.set_version)
    else:  # --bump is used
        uv_command.extend(["--bump", args.bump])
        if args.prerelease:
            uv_command.extend(["--bump", args.prerelease])

    uv_output = run_command(uv_command, dry_run=args.dry_run, capture_output=True)

    if args.dry_run:
        print_info("Dry run mode: Assuming new version is '1.2.3'.")
        new_version = "1.2.3"
    else:
        new_version = parse_new_version(uv_output)

    # 2. Create the git tag
    tag_name = f"v{new_version}"
    git_tag_command = ["git", "tag", tag_name]
    run_command(git_tag_command, dry_run=args.dry_run)

    # 3. Push the new tag to the remote
    if not args.no_push:
        git_push_command = ["git", "push", "origin", tag_name]
        run_command(git_push_command, dry_run=args.dry_run)
    else:
        print_info("Skipping git push due to --no-push flag.")

    print_success(f"Successfully tagged version {tag_name}.")
    if not args.no_push:
        print_success(
            "Tag pushed to remote. The GitHub Action should now trigger the publish process."
        )
    else:
        print_info(
            f"Remember to push the tag manually when ready: git push origin {tag_name}"
        )


if __name__ == "__main__":
    main()
