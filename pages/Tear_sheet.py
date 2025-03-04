st.write(pf.create_full_tear_sheet(
    returns=st.session_state.returns,
    transactions=st.session_state.transactions,
    live_start_date='2022-01-01',
    estimate_intraday=False,
    round_trips=False))
