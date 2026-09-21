def check_quote(diagnosis, quote, cost_min=None, cost_max=None):
    if quote is None:
        diagnosis.quote_verdict = "No Quote Provided"
        diagnosis.quote_reason = ""
        return diagnosis

    if cost_min is None or cost_max is None:
        diagnosis.quote_verdict = "Cannot Assess"
        diagnosis.quote_reason = (
            "The knowledge base does not contain a complete numeric "
            "cost range for this repair."
        )
        return diagnosis

    quote = float(quote)

    if quote <= cost_max:
        verdict = "Fair"
        reason = (
            f"The quoted amount of ₹{quote:.0f} is within or below "
            f"the KB range of ₹{cost_min:.0f}–₹{cost_max:.0f}."
        )

    elif quote <= cost_max * 1.5:
        verdict = "Potentially High"
        reason = (
            f"The quoted amount of ₹{quote:.0f} is above the KB upper "
            f"range of ₹{cost_max:.0f}."
        )

    else:
        verdict = "Potentially High"
        reason = (
            f"The quoted amount of ₹{quote:.0f} is substantially above "
            f"the KB upper range of ₹{cost_max:.0f}."
        )

    diagnosis.quote_verdict = verdict
    diagnosis.quote_reason = reason

    return diagnosis