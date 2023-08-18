def get_steps_to_accumulate_grad(
    simulated_batch_size: int, batch_size: int
) -> int:
    if simulated_batch_size is None:
        return 1
    if simulated_batch_size % batch_size != 0:
        raise ValueError(
            "Effective batch size must be a multiple of batch size! "
            f"{simulated_batch_size} % {batch_size} !=0"
        )
    return int(simulated_batch_size / batch_size)
