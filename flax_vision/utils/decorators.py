def add_start_doctring(*docstr):
    """
    Function inspired from https://github.com/huggingface/transformers/blob/dbd9641c8c0e146c078cbee11cdefcf556f6c817/src/transformers/utils/doc.py#L23
    """
    def docstr_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ or "")
        return fn

    return docstr_decorator
