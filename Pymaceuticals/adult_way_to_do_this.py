# %%
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

mouse_metadata_path = 'data/Mouse_metadata.csv'
study_results_path = 'data/Study_results.csv'

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)
combined_df = pd.merge(mouse_metadata, study_results)

# region

# region
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# endregion


def spaceless_lowers(dataframe):
    """
    :param dataframe: a dataframe with columns that have spaces and uppercase letters
    :return: a dataframe with the spaces replaced with _ and all caps made lowercase.
    """
    try:
        cols = dataframe.columns
        cols = [col.replace(' ', '_').lower() for col in cols]
        dataframe.columns = cols

        return dataframe

    except NameError:
        print('There is an unresolved reference to the dataframe in the function\'s argument.\n'
              'Make sure that the dataframe has been read and defined.')


mouse_cols_original = mouse_metadata.columns
study_cols_original = study_results.columns
combined_cols_original = combined_df.columns

df_list = [mouse_metadata, study_results, combined_df]

for df in df_list:
    df = spaceless_lowers(df)

combined_df.sex.replace(
        {
                'Female': 0,
                'Male': 1
        },
        inplace=True
)

# combined_df.drug_regimen = combined_df.drug_regimen.astype('category')
# endregion


# %%

mice = combined_df[combined_df.duplicated(subset=['mouse_id', 'timepoint'])]['mouse_id'].unique()
mice
# %%

# Checking the number of mice.
print(combined_df.mouse_id.nunique())

# %%

# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint.

# pycharm tells me the line below has no effect, but im including it for you guys
# because i know you like that
combined_df.drop(combined_df[~combined_df.mouse_id.isin(mice)].index)[0:10]
# %%

# Optional: Get all the data for the duplicate mouse ID.
optional_df = combined_df.drop(combined_df[~combined_df.mouse_id.isin(mice)].index)
print(optional_df)

# %%

# Create a clean DataFrame by dropping the duplicate mouse by its ID.

cleaned_df = combined_df.drop(combined_df[combined_df.mouse_id.isin(mice)].index)
# %%

# Checking the number of mice in the clean DataFrame.
print(cleaned_df.mouse_id.nunique())

# %%
# Generate a summary statistics table of mean, median, variance,
# standard deviation, and SEM of the tumor volume for each regimen
stats_to_get = 'mean median var std sem'.split()


# region
def the_mean(gbo):
    return gbo.mean()


def the_median(gbo):
    return gbo.median()


def the_var(gbo):
    return gbo.var()


def the_std(gbo):
    return gbo.std()


def the_sem(gbo):
    return gbo.sem()


# endregion

stats_table = cleaned_df.groupby('drug_regimen')

fcns = [the_mean, the_median, the_var, the_std, the_sem]
names = ['mean', 'median', 'var', 'stdev', 'sem']
names = [name.title() if name != 'sem' else name.upper() for name in names]
fcn_names = dict(zip(names, fcns))
# %%

the_long_way = {}

for k, v in fcn_names.items():
    the_long_way[k] = v(stats_table)['tumor_volume_(mm3)']

# are you gonna penalize me for rounding?
df_long_way = round(pd.DataFrame(the_long_way).rename_axis(mouse_cols_original[1]), 3)
print(df_long_way)

# %%


# Use groupby and summary statistical methods to calculate the following properties of each drug regimen:
# mean, median, variance, standard deviation, and SEM of the tumor volume.
# ! i already did
why_couldnt_i_do_this_earlier = stats_table.agg(stats_to_get)['tumor_volume_(mm3)']
why_couldnt_i_do_this_earlier.columns = [col.title() if col != 'sem' else col.upper() for col in
                                         why_couldnt_i_do_this_earlier.columns]
# Assemble the resulting series into a single summary dataframe.
print(why_couldnt_i_do_this_earlier)

# %%

# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for
# each regimen
# the default backend is matplotlib, so these next things are essentially the same.
# pd.options.plotting.backend = 'matplotlib'
# Using the aggregation method, produce the same summary statistics in a single line
# ! i already did this
# Generate a bar plot showing the total number of timepoints
# for all mice tested for each drug regimen using Pandas.

gbo_drugs = combined_df.groupby('drug_regimen')['mouse_id']

gbo_drugs.count().plot(
        kind='bar',
        title='Total Timepoints for all Mice in each Regimen',
        figsize=(10, 9),
        xlabel='Drug Used',
        ylabel='Count of Mice',
        rot=45
)

plt.savefig('gbo_drugs_pd.png')

plt.show()

# %%

# Generate a bar plot showing the total number of timepoints for all mice tested for each drug regimen using pyplot.
x, y = gbo_drugs.groups.keys(), gbo_drugs.count()

fig, ax = plt.subplots(figsize=(10, 9))

plot1 = ax.bar(
        x,
        height=y,
        label='Drug Regimen'
)

plt.xticks(rotation=45)

plt.xlabel('Drug Regimen')
plt.ylabel('Count of Mice')
plt.title('Total Timepoints for all Mice in each Regimen')

plt.tight_layout()
plt.show()
plt.savefig('gbo_drugs_plt.png')
# %%
gbo_sex = combined_df.groupby('sex')
fig, ax = plt.subplots()
# Generate a pie plot showing the distribution of female versus male mice using Pandas
gbo_sex.count().plot(
        kind='pie',
        y='mouse_id',
        title='Male vs Female Mice',
        ax=ax,
        ylabel='Count of Sex',
        figsize=(6, 6),
        labels=['Female', 'Male'],
        autopct='%2.1f%%'
)

ax.legend(['Female', 'Male'])
plt.tight_layout()
plt.savefig('gbo_sex_pd.png')
plt.show()
# %%

# Generate a pie plot showing the distribution of female versus male mice using pyplot
fig, ax = plt.subplots(figsize=(8, 8))
labels = ['Female', "Male"]

sizes = gbo_sex['sex'].count()

ax.pie(
        x=sizes,
        labels=labels,
        autopct='%3.1f%%',
        textprops={'fontsize': 14}
)

plt.legend()

plt.tight_layout()

plt.savefig('gbo_sex_plt.png')
plt.show()

# %%

# Calculate the final tumor volume of each mouse across four of the treatment regimens:
# Capomulin, Ramicane, Infubinol, and Ceftamin

best_drugs = 'Capomulin Ramicane Infubinol Ceftamin'.split()
best_drugs_df = combined_df[combined_df['drug_regimen'].isin(best_drugs)]

best_drugs_df['drug_regimen'].unique()
best_drugs_df.sort_values('timepoint', inplace=True)

# Start by getting the last (greatest) timepoint for each mouse
gbo_best_drugs = best_drugs_df.groupby(['drug_regimen', 'mouse_id']).last()['tumor_volume_(mm3)']

gbo_best_drugs.head(30)
# best_drugs_df = gbo_best_drugs.reset_index()

# %%
# concat_df = pd.concat([cleaned_df, best_drugs_df], join='inner', copy=False).drop_duplicates()
# concat_df = pd.merge(cleaned_df, best_drugs_df)
# cconc = concat_df[concat_df['drug_regimen'].isin(best_drugs)]
mdf = pd.merge(cleaned_df, best_drugs_df)
in_mdf = mdf[mdf['drug_regimen'].isin(best_drugs)]
# %%
tvol_gbo = in_mdf.groupby('drug_regimen')

quants = tvol_gbo['tumor_volume_(mm3)'].quantile([0.25, 0.75])

# %%
list_tvol = in_mdf.groupby('drug_regimen')['tumor_volume_(mm3)'].apply(list)
# list_tvol_dropped = list_tvol.dropna()
# %%
quant_gbo = quants.groupby('drug_regimen')
iqrs = quant_gbo.last() - quant_gbo.first()
iqrs

# %%
d = {}
dfs = {}
for drug in iqrs.index:
    lbound = quants[drug][0.25] - 1.5 * iqrs[drug]
    ubound = quants[drug][0.75] + 1.5 * iqrs[drug]

    d[drug] = [lbound, ubound]
    print(
            f'{drug}:\n'
            f'Lowerbound: {round(d[drug][0], 3)}\n'
            f'Upperbound: {round(d[drug][1], 3)}\n\n'
    )

    drug_df = in_mdf[in_mdf['drug_regimen'] == drug]

    between_df = drug_df[~drug_df['tumor_volume_(mm3)'].between(left=d[drug][0], right=d[drug][1])]
    dfs[drug] = between_df
    print(dfs[drug])

# %%
# ?Put treatments into a list for for loop (and later for plot labels)
# ?Create empty list to fill with tumor vol data (for plotting)
# ?Calculate the IQR and quantitatively determine if there are any potential outliers.

fig, ax = plt.subplots()

ax.boxplot(
        list_tvol,
        labels=best_drugs
)

ax.set_title('Tumor Volume Based on Drug Regimen')
ax.set_xlabel('Drug Regimen')
ax.set_ylabel('Tumor Volume (mm³)')

plt.savefig('boxplot.png')
plt.show()

# tumor_vol = the_merge_df.groupby('drug_regimen')['tumor_volume_(mm3)'].apply(list)
# tumor_vol = tumor_vol.reindex(best_drugs)
# quants = the_merge_df.groupby('drug_regimen_x')['tumor_volume_(mm3)_x'].quantile([0.25, 0.5, 0.75])
# tumor_vol_df = pd.DataFrame(tumor_vol)
# tumor_vol_df = tumor_vol_df.reindex(best_drugs)
# tumor_vol_df
# %%
# ?Locate the rows which contain mice on each drug and get the tumor volumes
# mice = tvol_gbo.mouse_id.sample(1, random_state=42)
# combined = list(zip(best_drugs, mice))
# data = {}
# sl = []
# for drug, mouse in combined:
#     data[drug] = in_mdf[in_mdf['mouse_id'] == mouse]['tumor_volume_(mm3)']
#     sl.append(pd.Series(data))
#     days = in_mdf.timepoint.unique()
#
# data.keys()



#%%
#? Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin
line_data = in_mdf[in_mdf.sort_values('mouse_id')['drug_regimen'] == 'Capomulin'][0:10]
line_data.plot(y='timepoint', x='tumor_volume_(mm3)')
plt.show()
# data_df = pd.DataFrame(list(data.keys()), columns=data.values())
# data_df
# data_df.dropna()
# data_df.columns = ['Drug Regimen', 'Tumor Volume']
# data_df
# %%

capo_mouse = in_mdf[in_mdf['drug_regimen'] == 'Capomulin']['mouse_id'].sample(1).item()
# capo_tumor_vol = in_mdf[in_mdf['mouse_id'] == capo_mouse]['tumor_volume_(mm3)']
# capo_days = in_mdf[in_mdf['mouse_id'] == capo_mouse]['timepoint']
plt.plot(
    days, data['Capomulin']
)

plt.title(f'Mouse: {capo_mouse}')

plt.xlabel('Timepoint (days)')
plt.ylabel('Tumor Volume (mm³)')
plt.show()
# add subset

# %%
# ? Generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen

capo_df = tvol_gbo.get_group('Capomulin').groupby('mouse_id').mean()

plt.scatter(
        capo_df['weight_(g)'], capo_df['tumor_volume_(mm3)']
)

plt.title('Average Tumor Volume vs Average Mouse Weight (Capomulin)')
plt.xlabel('Weight (g)')
plt.ylabel('Tumor Volume (mm³)')

plt.show()

# %%
correlation = capo_df[['weight_(g)', 'tumor_volume_(mm3)']].corr()
correlation = correlation.iloc[0][1]

print(
        f'The correlation between mean mouse weight and mean tumor volume is {round(correlation), 3}.'
)

reg = st.linregress(
        capo_df['weight_(g)'],
        capo_df['tumor_volume_(mm3)']
)
reg

#%%
