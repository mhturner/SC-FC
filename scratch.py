import svgutils.transform as sg
import os


data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC'


fig = sg.SVGFigure("8.5in", "11in")


fig1 = sg.fromfile(os.path.join(data_dir, 'figpanels', 'Fig8.svg'))
fig2 = sg.fromfile(os.path.join(data_dir, 'figpanels', 'Fig12.svg'))


fig1.getroot().moveto(0, 0, scale=1)
fig2.getroot().moveto(305, 0, scale=1)

# add text labels
txt1 = sg.TextElement(25,20, "A", size=12, weight="bold")
txt2 = sg.TextElement(305,20, "B", size=12, weight="bold")

# append plots and labels to figure
fig.append([fig1.getroot(), fig2.getroot()])
fig.append([txt1, txt2])

# save generated SVG files
fig.save(os.path.join(data_dir, 'svg_figs', 'Fig3.svg'))
