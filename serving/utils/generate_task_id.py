import random
import string
import time
from datetime import datetime


def generate_task_id():
    """
    Generate a random task ID in the format XXXX-XXXX-XXXX-XXXX-XXXX.

    Notes:
    - Mirrors LightX2V's task id format for API compatibility.
    - Does not modify the global random state.
    """
    original_state = random.getstate()
    try:
        characters = string.ascii_uppercase + string.digits
        local_random = random.Random(time.perf_counter_ns())

        groups = []
        for _ in range(5):
            time_mix = int(datetime.now().timestamp())
            local_random.seed(time_mix + local_random.getstate()[1][0] + time.perf_counter_ns())
            groups.append("".join(local_random.choices(characters, k=4)))

        return "-".join(groups)
    finally:
        random.setstate(original_state)

