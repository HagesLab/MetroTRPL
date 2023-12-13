import numpy as np
from sim_utils import MAX_PROPOSALS

def approve_move(new_state, shared_fields):
    """ Raise a warning for non-physical or unrealistic proposed trial moves,
        or proposed moves that exceed the prior distribution.
    """
    order = shared_fields['names']
    checks = {}
    prior_dist = shared_fields["prior_dist"]

    # Ensure proposal stays within bounds of prior distribution
    diff = np.where(shared_fields["do_log"], 10 ** new_state, new_state)
    for i, param in enumerate(order):
        if not shared_fields["active"][i]:
            continue

        lb = prior_dist[param][0]
        ub = prior_dist[param][1]
        checks[f"{param}_size"] = (lb < diff[i] < ub)

    # TRPL specific checks:
    # p0 > n0 by definition of a p-doped material
    if 'p0' in order and 'n0' in order:
        checks["p0_greater"] = (new_state[shared_fields["_param_indexes"]["p0"]]
                                > new_state[shared_fields["_param_indexes"]["n0"]])
    else:
        checks["p0_greater"] = True

    # tau_n and tau_p must be *close* (within 2 OM) for a reasonable midgap SRH
    if 'tauN' in order and 'tauP' in order:
        # Compel logscale for this one - makes for easier check
        logtn = new_state[shared_fields["_param_indexes"]['tauN']]
        if not shared_fields["do_log"][shared_fields["_param_indexes"]["tauN"]]:
            logtn = np.log10(logtn)

        logtp = new_state[shared_fields["_param_indexes"]['tauP']]
        if not shared_fields["do_log"][shared_fields["_param_indexes"]["tauP"]]:
            logtp = np.log10(logtp)

        diff = np.abs(logtn - logtp)
        checks["tn_tp_close"] = (diff <= 2)

    else:
        checks["tn_tp_close"] = True

    failed_checks = [k for k in checks if not checks[k]]

    return failed_checks

def make_trial_move(current_state, trial_move, shared_fields, RNG, logger):
    """ 
    Trial move function: returns a new proposed state equal to the current_state plus a uniform random displacement
    """

    _current_state = np.array(current_state, dtype=float)

    mu_constraint = shared_fields.get("do_mu_constraint", None)

    _current_state = np.where(shared_fields["do_log"],
                                np.log10(_current_state),
                                _current_state)

    tries = 0

    # Try up to MAX_PROPOSALS times to come up with a proposal that stays within
    # the hard boundaries, if we ask
    if shared_fields.get("hard_bounds", 0):
        max_tries = MAX_PROPOSALS
    else:
        max_tries = 1

    new_state = np.array(_current_state)
    while tries < max_tries:
        tries += 1

        new_state = _current_state + trial_move * (2 * RNG.random(_current_state.shape) - 1)

        if mu_constraint is not None:
            ambi = mu_constraint[0]
            ambi_std = mu_constraint[1]
            logger.debug(f"mu constraint: ambi {ambi} +/- {ambi_std}")
            new_muambi = np.random.uniform(ambi - ambi_std, ambi + ambi_std)
            new_state[shared_fields["_param_indexes"]["mu_p"]] = np.log10(
                (2 / new_muambi - 1 / 10 ** new_state[shared_fields["_param_indexes"]["mu_n"]])**-1)

        failed_checks = approve_move(new_state, shared_fields)
        success = len(failed_checks) == 0
        if success:
            logger.debug(f"Found params in {tries} tries")
            break

        if len(failed_checks) > 0:
            logger.warning(f"Failed checks: {failed_checks}")

    new_state = np.where(shared_fields["do_log"], 10 ** new_state, new_state)
    return new_state