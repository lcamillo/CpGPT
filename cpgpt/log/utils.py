import sys
from pathlib import Path

from loguru import logger


def get_class_logger(cls: type, log_dir: str = "logs") -> logger:
    """Returns a logger object for the given class.

    Args:
        cls: The class for which the logger is being created.
        log_dir: The directory where log files will be stored. Defaults to "logs".

    Returns:
        The logger object for the given class.

    """
    class_name = cls.__name__
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<bold><blue>cpgpt</blue></bold> -"
            "<cyan>{extra[class_name]}</cyan>: "
            "<level>{message}</level>"
        ),
        level="INFO",
    )

    # Ensure the log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_file = Path(log_dir) / f"{class_name.lower()}.log"

    logger.add(
        log_file,
        rotation="10 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[class_name]}: {message}",
    )

    class_logger = logger.bind(class_name=class_name)

    # Log the initialization message
    class_logger.info(f"Initializing class {class_name}.")

    return class_logger


class DownloadProgressBar:
    """A progress bar for tracking the download progress of a file.

    Args:
        logger (Logger): The logger object to use for logging.
        class_name (str): The name of the class using the progress bar.
        total (Optional[int]): The total size of the file being downloaded. Defaults to None.
        unit (str): The unit of the file size. Defaults to 'B'.
        unit_scale (bool): Whether to scale the file size units. Defaults to True.
        unit_divisor (int): The divisor to use for scaling the file size units. Defaults to 1024.
        ncols (int): The width of the progress bar in characters. Defaults to 50.

    Methods:
        update(n: int) -> None:
            Updates the progress bar with the given value.
        display() -> None:
            Displays the progress bar.
        format_size(size) -> str:
            Formats the given size in bytes to a human-readable string.
        close() -> None:
            Closes the progress bar.
    Usage:
        with DownloadProgressBar(logger, class_name, total) as progress_bar:
            progress_bar.update(n)

    """

    def __init__(
        self,
        logger: logger,
        class_name: str,
        total: int | None = None,
        unit: str = "B",
        unit_scale: bool = True,
        unit_divisor: int = 1024,
        ncols: int = 30,
    ) -> None:
        """Initialize the DownloadProgressBar.

        Args:
            logger: Logger object for logging progress.
            class_name: Name of the class using the progress bar.
            total: Total size of the file being downloaded.
            unit: Unit of the file size (default: 'B').
            unit_scale: Whether to scale the file size units (default: True).
            unit_divisor: Divisor for scaling file size units (default: 1024).
            ncols: Width of the progress bar in characters (default: 30).

        """
        self.logger = logger
        self.class_name = class_name
        self.total = total
        self.n = 0
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.ncols = ncols
        self.last_msg = ""

    def update(self, n: int) -> None:
        """Updates the value of `n` by adding the given integer `n` to it."""
        self.n += n
        self.display()

    def display(self) -> None:
        """Displays the progress bar."""
        percentage = min(100, self.n / self.total * 100) if self.total is not None else 0
        filled_length = int(self.ncols * percentage // 100)
        bar = "█" * filled_length + "-" * (self.ncols - filled_length)

        n_fmt = self.format_size(self.n)
        total_fmt = self.format_size(self.total) if self.total is not None else "Unknown"

        progress_str = f"{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
        message = (
            f"\033[1m\033[34mcpgpt\033[0m: \033[36m{self.class_name}\033[0m: "
            f"\033[1m{progress_str}\033[0m\r"
        )

        sys.stdout.write("\r" + " " * len(self.last_msg) + "\r")  # Clear the last message
        sys.stdout.write(message)
        sys.stdout.flush()
        self.last_msg = message

    def format_size(self, size: int | None) -> str:
        """Formats the given size in bytes to a human-readable string."""
        if size is None:
            return "Unknown"
        if self.unit_scale:
            size /= self.unit_divisor
        for unit in ["K", "M", "G", "T", "P"]:
            if size < 1024:
                return f"{size:.2f}{unit}{self.unit}"
            size /= 1024
        return f"{size:.2f}P{self.unit}"

    def close(self) -> None:
        """Closes the progress bar."""
        sys.stdout.write("\n")
        sys.stdout.flush()

    def __enter__(self) -> "DownloadProgressBar":
        """Enters the context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exits the context manager."""
        self.close()
