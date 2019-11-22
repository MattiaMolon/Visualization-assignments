import wx
from gui.application import GLFrame
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except:
    pass

app = wx.App()
frame = GLFrame(None, "Visualizer")
frame.Show()

app.MainLoop()
app.Destroy()
