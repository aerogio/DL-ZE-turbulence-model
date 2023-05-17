# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'OpenFOAMReader'

folder = '/NOBACKUP2/gcal/openfoam/Room-Block-Heat/velocity-cases/'

cases = ['0.5','1','1.366','2','5','10']

folder2 = '/NOBACKUP2/gcal/ANNs/MLPs/Slice-Room-RNG-vel8/Data'

A = 0.8128
B = 1.6256
M = 1.2192
# points = [M]
points = [A,M,B]

for case in cases:
    print('--> STARTING CASE {} \n'.format(case))
    animationScene1 = GetAnimationScene()
    animationScene1.GoToLast()
    kfoam = OpenFOAMReader(FileName='{}/{}/k.foam'.format(folder,case))
    kfoam.CellArrays = ['U', 'nut', 'yWall']
    kfoam.MeshRegions = ['internalMesh']

    for point in points:
        
        slice1 = Slice(Input=kfoam)
        slice1.SliceType = 'Plane'
        slice1.SliceOffsetValues = [0.0]
        slice1.SliceType.Origin = [1.2191562970037921, 1.2191999104634306, point]
        slice1.SliceType.Normal = [0.0, 0.0, 1.0]
        CreateLayout('Layout #2')

        # set active view
        SetActiveView(None)
        spreadSheetView1 = CreateView('SpreadSheetView')
        layout2 = GetLayout()
        spreadSheetView1.Update()
        SetActiveView(spreadSheetView1)
        animationScene1 = GetAnimationScene()
        animationScene1.GoToLast()
        # show data in view
        layout2.AssignView(0, spreadSheetView1)

        slice1Display_1 = Show(slice1, spreadSheetView1)

        ExportView('{}/d-{}-{}.csv'.format(folder2, case, point), view=spreadSheetView1)
        Delete(slice1)
        del slice1

        SetActiveSource(kfoam)


# ResetSession()

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
