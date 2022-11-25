import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import numpy as np
import os

metrics = ['node_perf_cpu_migrations_total', 'node_perf_branch_misses_total',
           'node_perf_context_switches_total', 'node_perf_cache_bpu_read_misses_total']
os.popen(
    "kubectl get node -o wide|grep -v 'NAME'|awk '{print $1,$6}'> node.t>node.t")
columns = ['timestamp', 'value', 'cpu', 'metrics', 'node']
columns_all = ['timestamp', 'value', 'metrics']
perf_columns = ['time', 'count', 'unit', 'events']


def evict_port(x):
    return x.split(':')[0]


def trim_num(x):
    return float("{:.3f}".format(float(x)))


def reset_time(x, s):
    return x-s


def get_perf_total(metric_name='node_perf_cpu_migrations_total', start=1629342630, end=1629343530, step=15):
    headers = {
        "Authorization": "Bearer eyJrIjoiTlVWSzJjWjEyY2NiSVJuU2hVRXBTMXhMWFZSbEFWNUoiLCJuIjoiZmZmZiIsImlkIjoxfQ=="}
    url = 'http://172.16.101.6:30965/api/datasources/proxy/1/api/v1/query_range?query={metric}&start={start}&end={end}&step={step}'
    metric = 'sum(rate({mtr}'.format(mtr=metric_name) + '{}[20s]))'
    url = url.format(metric=metric, start=start, end=end, step=step)
    print(url)
    r = requests.get(url, headers=headers)
    df = r.json()['data']['result']
    df = pd.json_normalize(df)
    dk = pd.DataFrame(columns=columns_all)
    for i, r in df.iterrows():
        df_values = pd.DataFrame(r['values'], columns=['timestamp', 'value'])
        start_time = df_values['timestamp'][0]
        df_values['metrics'] = metric_name
        df_values['timestamp'] = df_values.apply(
            lambda x: reset_time(x.timestamp, start_time), axis=1)
        dk = dk.append(df_values)
    dk['value'] = dk.apply(lambda x: trim_num(x.value), axis=1)

    return dk


def get_perf(metric_name='node_perf_cpu_migrations_total', start=1629342630, end=1629343530, step=15):
    headers = {
        "Authorization": "Bearer eyJrIjoiTlVWSzJjWjEyY2NiSVJuU2hVRXBTMXhMWFZSbEFWNUoiLCJuIjoiZmZmZiIsImlkIjoxfQ=="}
    url = 'http://172.16.101.6:30965/api/datasources/proxy/1/api/v1/query_range?query={metric}&start={start}&end={end}&step={step}'
    metric = 'sum(rate({mtr}'.format(mtr=metric_name) + \
        '{}[20s])) by (cpu,instance)'
    url = url.format(metric=metric, start=start, end=end, step=step)
    print(url)
    r = requests.get(url, headers=headers)
    df = r.json()['data']['result']
    df = pd.json_normalize(df)
    dk = pd.DataFrame(columns=columns)
    for i, r in df.iterrows():
        df_values = pd.DataFrame(r['values'], columns=['timestamp', 'value'])
        start_time = df_values['timestamp'][0]
        df_values['cpu'] = r['metric.cpu']
        df_values['node'] = r['metric.instance']
        df_values['metrics'] = metric_name
        df_values['timestamp'] = df_values.apply(
            lambda x: reset_time(x.timestamp, start_time), axis=1)
        dk = dk.append(df_values)
    dk['node'] = dk.apply(lambda x: evict_port(x.node), axis=1)
    dk['value'] = dk.apply(lambda x: trim_num(x.value), axis=1)
    return dk


def perf_2_csv_total(start, end, platform, header=False):
    start = start-30
    end = end + 30
    perf_total_df = pd.DataFrame(columns=columns_all)
    for m in metrics:
        dk = get_perf_total(metric_name=m, start=start, end=end)
        perf_total_df = perf_total_df.append(dk)
    perf_total_df['platform'] = platform
    perf_total_df.to_csv('perf_total_df.csv', mode='a',
                         index=False, header=header)


perf_total_df = pd.read_csv('perf_total_df.csv')
perf_total_df


start_ow = 1630138190
end_ow = 1630138371
interval = end_ow-start_ow
platform = 'openwhisk'
perf_2_csv_total(start_ow, end_ow, platform, header=True)

start_of = 1630138692
end_of = 1630138707
platform = 'openfaas'
plot_end_of = start_of+interval
perf_2_csv_total(start_of, plot_end_of, platform)


start_kl = 1630138802
end_kl = 1630138821
plot_end_kl = start_kl+interval
platform = 'kubeless'
perf_2_csv_total(start_kl, plot_end_kl, platform)


start_none = 1630138921
end_none = start_none+interval
platform = 'none'
perf_2_csv_total(start_none, end_none, platform)

# matplotlib inline
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(26, 4))

# palette=sns.cubehelix_palette(4,
#     start=0.00,
#     rot=1.00,
#     gamma=0.60,
#     hue=0.80,
#     light=0.3,
#     dark=0,
#     reverse=True,
#     as_cmap=False)
palette = sns.cubehelix_palette(
    n_colors=4, start=2.10, rot=1.00, gamma=2.00, hue=0.40, light=0.80, dark=0.40, reverse=True)
# perf_total_df = perf_total_df.loc[perf_total_df.platform=='kubelss']


node_perf_cpu_migrations_total = perf_total_df.loc[(
    perf_total_df.metrics == 'node_perf_cpu_migrations_total')]
node_perf_branch_misses_total = perf_total_df.loc[(
    perf_total_df.metrics == 'node_perf_branch_misses_total')]
node_perf_context_switches_total = perf_total_df.loc[(
    perf_total_df.metrics == 'node_perf_context_switches_total')]
node_perf_cache_bpu_read_misses_total = perf_total_df.loc[(
    perf_total_df.metrics == 'node_perf_cache_bpu_read_misses_total')]

# axes[0].set_yscale('log')
# axes[1].set_yscale('log')
# axes[2].set_yscale('log')
# axes[3].set_yscale('log')

# axes[0].set_xlable('log')
# axes[1].set_yscale('log')
# axes[2].set_yscale('log')
# axes[3].set_yscale('log')

sns.lineplot(ax=axes[0], x='timestamp', y="value",
             palette=palette, hue="platform",
             data=node_perf_cpu_migrations_total)

sns.lineplot(ax=axes[1], x='timestamp', y="value",
             palette=palette, hue="platform",
             data=node_perf_context_switches_total)

sns.lineplot(ax=axes[2], x='timestamp', y="value",
             palette=palette, hue="platform",
             data=node_perf_cache_bpu_read_misses_total)

sns.lineplot(ax=axes[3], x='timestamp', y="value",
             palette=palette, hue="platform",
             data=node_perf_branch_misses_total)

sns.choose_cubehelix_palette()


# bash perf.sh 'LLC-load-misses,branch-misses' 500 openwhisk
# bash perf_stop_get_data.sh
# tar -zxvf perf/172.tar.gz

df_perf = pd.read_csv('perf/home/tmp/cycles_172_hello-java-66ddd95588-m6svb_1630248661.808489808.csv',
                      delim_whitespace=True, header=None, skiprows=[0, 1, 2])
df_perf.columns = perf_columns
df_perf


params = 'cycles_172_hello-java-66ddd95588-m6svb_1630248661.808489808.csv'.split(
    '_')
params
perf_start = float(params[2].replace('.csv', ''))
perf_start

df_perf['node'] = params[0]
df_perf['timestamp'] = df_perf['time']+perf_start
df_perf['functions'] = params[1]
df_perf
