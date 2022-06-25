# from https://github.com/chrispaulca/waterfall

import numpy as np
import pandas as pd
import matplotlib.pyplot as pls

def plot(
    index, 
    data,
    ax=None,
    formatting='{:,.1f}', 
    green_color='#29EA38', 
    red_color='#FB3C62', 
    blue_color='#24CAFF',
    sorted_value=False, 
    threshold=None, 
    other_label='other', 
    net_label='net', 
    rotation=45, 
    blank_color=(0,0,0,0), 
    fontsize=None,
    line_kwargs={},
    bar_kwargs={}
):
    '''
    Given two sequences ordered appropriately, generate a standard waterfall chart.
    Optionally modify number formatting, bar colors, 
    increment sorting, and thresholding. Thresholding groups lower magnitude changes
    into a combined group to display as a single entity on the chart.
    '''
    
    if ax is None:
        ax = plt.gca()
    
    if fontsize is None:
        fontsize = plt.rcParams['font.size'] * 0.9
    
    # convert data and index to np.array
    index = np.array(index)
    data = np.array(data)
    
    # sorted by absolute value 
    if sorted_value: 
        abs_data = abs(data)
        data_order = np.argsort(abs_data)[::-1]
        data = data[data_order]
        index = index[data_order]
    
    # group contributors less than the threshold into 'other' 
    if threshold:
        abs_data = abs(data)
        threshold_v = abs_data.max() * threshold
        
        if threshold_v > abs_data.min():
            index = np.append(index[abs_data >= threshold_v], other_label)
            data = np.append(data[abs_data >= threshold_v], sum(data[abs_data<threshold_v]))
    
    changes = {'amount': data}
    
    # Store data and create a blank series to use for the waterfall
    trans = pd.DataFrame(data=changes, index=index)
    blank = trans['amount'].cumsum().shift(1).fillna(0)
    
    trans['positive'] = (trans['amount'] > 0).astype(int)
    
    # Get the net total number for the final element in the waterfall
    total = trans.sum()['amount']
    trans.loc[net_label, ['amount', 'positive']] = [total, 99]
    blank.loc[net_label] = total
    
    # The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan
    
    # When plotting the last element, we want to show the full bar,
    # Set the blank to 0
    blank.loc[net_label] = 0
    
    trans['color'] = trans['positive']
    trans.loc[trans['positive'] == 1, 'color'] = green_color
    trans.loc[trans['positive'] == 0, 'color'] = red_color
    trans.loc[trans['positive'] == 99, 'color'] = blue_color
    
    my_colors = list(trans['color'])
    
    # Plot and label
    ax.bar(range(0, len(trans.index)), blank, width=0.5, color=blank_color, **bar_kwargs)
    ax.bar(range(0, len(trans.index)), trans['amount'], width=0.6, bottom=blank, color=my_colors, **bar_kwargs)
    
    # connecting lines - figure out later
    # my_plot = lines.Line2D(step.index, step.values, color = "gray")
    # my_plot = lines.Line2D((3,3), (4,4))
    
    # Get the y-axis position for the labels
    y_height = trans['amount'].cumsum().shift(1).fillna(0)
    
    temp = list(trans['amount'])
    
    # create dynamic chart range
    for i in range(len(temp)):
        if (i > 0) & (i < (len(temp) - 1)):
            temp[i] = temp[i] + temp[i-1]
    
    trans['temp'] = temp
    
    plot_max = trans['temp'].max()
    plot_min = trans['temp'].min()
    
    # Make sure the plot doesn't accidentally focus only on the changes in the data
    if all(i >= 0 for i in temp):
        plot_min = 0
    if all(i < 0 for i in temp):
        plot_max = 0
    
    if abs(plot_max) >= abs(plot_min):
        maxmax = abs(plot_max)   
    else:
        maxmax = abs(plot_min)
    
    pos_offset = maxmax / 40
    plot_offset = maxmax / 15 ## needs to me cumulative sum dynamic
    
    # Start label loop
    loop = 0
    for index, row in trans.iterrows():
        # For the last item in the list, we don't want to double count
        if row['amount'] == total:
            y = y_height[loop]
        else:
            y = y_height[loop] + row['amount']
        # Determine if we want a neg or pos offset
        if row['amount'] > 0:
            y += (pos_offset*2)
            ax.annotate(formatting.format(row['amount']), (loop, y), ha='center', color=green_color, fontsize=fontsize)
        else:
            y -= (pos_offset*4)
            ax.annotate(formatting.format(row['amount']), (loop, y), ha='center', color=red_color, fontsize=fontsize)
        loop += 1
    
    # Scale up the y axis so there is room for the labels
    ax.set_ylim(plot_min-round(3.6*plot_offset, 7), plot_max+round(3.6*plot_offset, 7))
    
    # Rotate the labels
    ax.set_xticks(range(len(trans.index)))
    ax.set_xticklabels(trans.index, rotation=rotation, ha=('center' if rotation==90 else 'right'))
    
    #add zero line and title
    ax.axhline(0, color='black', **(dict(linewidth=0.6, linestyle='dashed') | line_kwargs))
    
    return ax
