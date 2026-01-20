import sys

def error_message_detail(error,detail:sys):
    _,_,exe_tb = detail.exc_info()
    file_name = exe_tb.tb_frame.f_code.co_filename
    line_number = exe_tb.tb_lineno
    error_msg = "Error occured in python script name [{0}] line number [{1}] and error message[{2}]".format(file_name,line_number,str(error))
    return error_msg

class CustomException(Exception):
    def __init__(self,error_msg,detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_message_detail(error_msg,detail)
    
    def __str__(self):
        return self.error_msg
    

# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         raise CustomException(e,sys)
    