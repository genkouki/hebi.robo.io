import sys



# from computer_vision import detect_blue_mark

argvs = sys.argv

def GUI():
    from Arm import pygame_GUI
    pyGUI = pygame_GUI.pygameGUI()
    while True:
        pyGUI.step()

def collect_data():
    from data_collect import data_collect
    collect = data_collect.DataCollect()
    if len(argvs)==3:
        if argvs[2] == 'clear':
            collect.clear_csv()
    if len(argvs)==2:
        while True:
            try:
                (a, b) = input("Please input 2 value or enter to exit: ").split()
                collect.write_data([int(a), int(b)])
            except:
                break
        collect.save_dataframe()
        print(collect.log_df.head())
        print("Done")
        
if len(argvs)>1:
    if argvs[1] == 'gui':
        GUI()

    if argvs[1] == 'collect_data':
        collect_data()

def main(has_cam=False,is_collecting_data=False):
    import threading
    from Arm import pygame_GUI
    from data_collect import data_collect
    if has_cam:
        from computer_vision import realsense_depth, detect_blue_mark
    from data_collect.model import Model
    import pathlib
    import numpy as np
    import matplotlib.pyplot as plt

    if is_collecting_data:
        class mainGUI(pygame_GUI.pygameGUI):
            def __init__(self, w=640, h=480) -> None:
                super().__init__(w, h)

            #rewrite func for button command
            def on_button_pressed(self, button):
                super().on_button_pressed(button)
                if button.name == 'button_b':
                    self.controller.set_rumble(0.7, 0.7, 300)
                    print(type(self.robot.group_fbk.position)) #takamori comment out
                    a = 2
                    self.blue_moments=detect_blue_mark.main(a) #kato edit
                    #print("stop")
                    self.data=[self.robot.group_fbk.position,self.blue_moments]#takamori wrote code 2022.12.15
                    #collect.write_data_to_csv(self.robot.group_fbk.position) #takamori comment out 2022.12.15
                    #collect.write_data_to_csv(self.blue_moments)#kato edit#,self.blue_moments)#takamori add code ",blue_moments" #takamori comment out 2022.12.15
                    collect.takamori_csv(self.data) #takamori wrote code 2022.12.15
                    #self.pos = np.array([ 0.04162088, 0.96691024, -1.98562271, 1.20379448, 0.88048851])
                    #pyGUI.set_position(self.pos)
        a = 1
        x = detect_blue_mark.main(a) #takamori comment out
        pyGUI = mainGUI() #kato comment out
        collect = data_collect.DataCollect(cols=pyGUI.robot.names)#kato
        
        # camera_thread = threading.Thread(target=detect_blue_mark.main, args=())
        # camera_thread.start()
        # rc = realsense_depth.DepthCamera()
        csvfile = pathlib.Path().absolute()/collect.csv_filename#kato
        end_effector = np.zeros(3)#kato
        for i in range(3):#kato
            end_effector[i]=pyGUI.robot.joint_angles[i]#kato
        # # # plt.ion()
        model = Model(end_effector=end_effector)#kato
        
        model.train(csv_file=csvfile, iteration=10, show_plot=True)#kato
        # pyGUI.robot.input_model(model)
        plt.show()#kato
        while True:#kato
            pyGUI.step()#kato
    else:
        gui = pygame_GUI.pygameGUI()
        while True:
            gui.step()

if len(argvs)==1 and __name__=='__main__':
    #camera connect --> has_cam=True
    main(has_cam=True,is_collecting_data=True)nck_581998