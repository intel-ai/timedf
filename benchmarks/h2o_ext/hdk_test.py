import modin.config as cfg
cfg.StorageFormat.put('hdk')

import modin.pandas as pd
import numpy as np



N = 10_0_000

data = np.arange(N)

x = pd.DataFrame({
    'id1': ['id' + str(d) for d in (data % 9)],
    'id2': ['id' + str(d) for d in (data % 5)],
    'id3': ['id' + str(d) for d in (data % 17)],
    'id4': data % 3,
    'id5': data % 13,
    'id6': data % 2,
    'v1': data,
    'v3': (data + 2).astype(np.float64) / 18,
})

ans = x.groupby(
        ["id1", "id2", "id3", "id4", "id5", "id6"],
    ).agg({"v3": "sum", "v1": "size"})
print(ans.shape)
ans['v3'].sum()
ans._query_compiler._modin_frame._execute()