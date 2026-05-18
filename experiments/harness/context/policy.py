from enum import Enum, auto

class OverflowPolicy(Enum):
    """
    Policy for handling context overflow
    when token budgets are exceeded.
    """

    # -----------------------------------
    # Basic token handling strategies
    # -----------------------------------
    COMPRESS = auto()               # Automatically summarize/compress memory when limits are exceeded
    WARN = auto()                   # Notify the system but do not change memory automatically
    ERROR = auto()                  # Fail immediately when limits are exceeded
    TRUNCATE = auto()               # Drop old content without semantic compression
    
    # -----------------------------------
    # Advanced context management strategies
    # -----------------------------------
    EVICT_LOW_PRIORITY = auto()     # Remove blocks with lowest compression priority first
    EVICT_LOW_IMPORTANCE = auto()   # Remove least valuable information first
    CHECKPOINT_AND_RESET = auto()   # Save memory to durable storage, then clear active context
    ISOLATE_UNTRUSTED = auto()      # Move unsafe tool outputs into isolated memory instead of prompt context