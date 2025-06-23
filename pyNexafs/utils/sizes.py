"""
Utility functions for handling sizes and conversions.
"""


def btyes_to_human_readable(num_bytes: int) -> str:
    """
    Convert bytes to a human-readable format (KB, MB, GB, etc.).

    Parameters
    ----------
    num_bytes : int
        Number of bytes.

    Returns
    -------
    str
        A human-readable string.
    """
    if num_bytes < 0:
        raise ValueError("Number of bytes must be non-negative.")

    if num_bytes == 0:
        return "0 B"

    size_units = ["B", "KB", "MB", "GB", "TB"]
    index = 0

    while num_bytes >= 1024 and index < len(size_units) - 1:
        num_bytes = num_bytes // 1024
        index += 1

    return f"{num_bytes:.2f} {size_units[index]}"


if __name__ == "__main__":
    # Example usage
    print(btyes_to_human_readable(1024))  # Output: "1.00 KB"
    print(btyes_to_human_readable(1048576))  # Output: "1.00 MB"
    print(btyes_to_human_readable(1073741824))  # Output: "1.00 GB"
    print(btyes_to_human_readable(0))  # Output: "0 B"
    print(btyes_to_human_readable(-1))  # Raises ValueError
