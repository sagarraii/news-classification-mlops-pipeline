import sys

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        f"Error occured in python script [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{str(error)}]"
    )

    return error_message

class CustomException(Exception):

    def __init__(self, error_message, error_detail):
        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message,
            error_detail
        )

    def __str__(self):
        return self.error_message
    

# Testing the exception
'''
if __name__ == "__main__":

    try:
        print("Starting test...")

        a = 10
        b = 0

        result = a / b

        print(result)

    except Exception as e:
        raise CustomException(e, sys)

 # note : test this file by running command: python src\exception\custom_exception.py in bash     
'''