class Logfile:
    def __init__(self, DIR):
        self.DIR = DIR
        self.log_file = open(DIR,'a')    
        
    def open(self):
        self.log_file = open(self.DIR,'a')
        
    def write(self,string):
        print(string,end="")        
        self.log_file.write(string)
        
    def close(self):
        self.log_file.close()