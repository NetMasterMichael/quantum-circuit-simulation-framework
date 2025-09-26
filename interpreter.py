from quantum_circuit import Circuit
import sys

class Interpreter:

    DIRECT_EXECUTE_ARGUMENT = "-e"

    def __init__(self):
        # Read program in
        direct_execution_mode = False
        file_execution_mode = False
        arguments = sys.argv[1:]
        if (arguments[0] == self.DIRECT_EXECUTE_ARGUMENT):
            direct_execution_mode = True
            program = arguments[1].split("\n")
        else:
            file_execution_mode = True
            with open(arguments[0], "r") as file:
                contents = file.read()
            program = contents.split("\n")
        
        # Load program into a stack
        self._execution_stack = []
        for line in program:
            self._execution_stack.append(line)
        
        # Start main interpreter loop
        self.main()


    def main(self):
        print(self._execution_stack)

if __name__ == "__main__":
    Interpreter()