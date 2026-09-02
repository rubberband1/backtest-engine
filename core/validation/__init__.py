"""Statistical validation of a backtest result.

Everything in here exists to answer one question: is the number produced by
`core.engine` an edge, or the residue of having looked at the same data many
times? Each module attacks it from a different side - out-of-sample decay,
comparison against a null model, and the correction owed for the number of
attempts made.

None of these modules sends orders or touches the broker. They read runs and
bars, and they report with the observation count attached to every figure.
"""
