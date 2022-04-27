from pyirr import rel_inter_intra


def test_rel_inter_intra(gonio):
    reliability = rel_inter_intra(gonio, nraters=2)
    expected = "{'Fs': {'F_inter': 17.3735,\n        'F_intra': 14.0485,\n        'F_intra_part': array([16.3533, 12" \
               ".3131])},\n 'lowvalue': {'low_intra': 1,\n              'low_intra_part': array([1., 1.]),\n        " \
               "      'lowinter_fixed': 0.9354,\n              'lowinter_random': 0.8536},\n 'pvalue': {'F_inter_p':" \
               " 0.0,\n            'F_intra_p': 0.0,\n            'F_intra_part_p': array([0., 0.])},\n 'rohat': {'r" \
               "hohat_inter_fixed': 0.9613,\n           'rhohat_inter_random': 0.9451,\n           'rhohat_intra_fix" \
               "ed': 0.984,\n           'rhohat_intra_fixed_part': array([0.9862, 0.9818]),\n           'rhohat_intr" \
               "a_random': 0.9842,\n           'rhohat_intra_random_part': array([0.9864, 0.9821])},\n 'sem ': {'SEM" \
               "inter_fixed': 1.4392,\n          'SEMinter_random': 1.7282,\n          'SEMintra': 0.9254,\n        " \
               "  'SEMintra_part': array([0.8577, 0.9884])}}"

    assert reliability.subjects == 29
    assert reliability.raters == 2
    assert reliability.detail == expected
