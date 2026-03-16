import pandas as pd


def test_leverage_computation_shape():
    df = pd.DataFrame({"champ_prob": [0.2], "field_pick": [0.1]})
    df["lev"] = df["champ_prob"] / df["field_pick"]
    assert df["lev"].iloc[0] == 2.0
