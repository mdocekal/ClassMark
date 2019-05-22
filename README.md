# ClassMark [cs]
Benchmark pro klasifikátory, který byl vytvořen jako součást diplomové práce Martinem Dočekalem na VYSOKÉM UČENÍ TECHNICKÉM V BRNĚ v roce 2019.
## Instalace
<i>(Popis instalace používá příkazy, které jsou napsány pro systémy jako je ubuntu/debian.)</i>

Tento benchmark je implementován v python 3, je tedy nutné jej mít nainstalovaný na vašem počítači.  Instalační skripty používají pip3. Ujistěte se tedy, že jej máte také nainstalovaný. Pro instalaci pip3 spusťte:
	
	apt install python3-pip

Classmark používá instalovatelné zásuvné moduly. Základní sada zásuvných modulů je ve složce plugins. Jeden ze základních zásuvných modulů (ANN klasifikátor) potřebuje pro svoji práci tensorflow. Preferovaná verze tensorflow je tensorflow-gpu která potřebuje CUDA. Jak nainstalovat CUDA může nalézt na: [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu). Pokud nechcete či nemůžete použít CUDA, pak můžete instalaci CUDA vynechat, ale použijte instalační skripty s koncovkou _CPU (popsáno níže).

Pro instalace ClassMark se základní sadou zásuvných modulů použijte:

	./install.sh

Jsou tu však i jiné instalační skripty, které můžete místo tohoto použít:

* install_editable.sh
  * Stejné jako install.sh, ale pro vývojové účely (pip3 install --editable).
* install_CPU.sh
  * Stejné jako install.sh, ale ANN klasifikátor použije tensorflow bez GPU (povolení GPU parametru v GUI nemá vliv). 
  * ./install.sh je preferovaná možnost. Prosím použijte ji je-li to možné.
* install_editable_CPU.sh
  * Stejné jako install_editable.sh, ale ANN klasifikátor použije tensorflow bez GPU (povolení GPU parametru v GUI nemá vliv). 
  * install_editable.sh je preferovaná možnost. Prosím použijte ji je-li to možné.

Chcete-li instalovat ClassMark pro ostatní systémy (např. Windows), pak se prosím podívejte do install*.sh skriptů pro získání inspirace. Skládají se pouze z několika pip3 příkazů a z jejich obsahu je hned jasné co je potřeba nainstalovat.


Na závěr je vhodné provést restart. Po restartu by mělo jít spustit ClassMark pomocí:

	classmark

## Příklady
Ve složce examples jsou příklady souborů s datovými sadami.

# ClassMark [en]
Benchmark for classifiers that was created as a part of diploma thesis by Martin Dočekal at BRNO UNIVERSITY OF TECHNOLOGY in 2019. 
## Installation
<i>(Installation description uses commands that are written for ubuntu/debian like systems.)</i>

This benchmark is implemented in python 3 so you must have it installed on your computer. Installation scripts are using the pip3, so make sure that this is also installed. For pip3 installation run:
	
	apt install python3-pip

ClassMark uses installable plugins. Basic set of plugins is in plugin folder. One of basic plugins (ANN classifier) needs for its work tensorflow. Preferable version of tensorflow is tensorflow-gpu, that needs CUDA. How to install CUDA could be found on: [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu). If you do not want to or can not use CUDA, than feel free to omit this step, but use installation script with _CPU suffix (described later).

For installation of ClassMark with basic set of plugins please use:

	./install.sh

There are also another types of installation scripts, that you can use instead:

* install_editable.sh
  * Same as install.sh, but for development purposes (pip3 install --editable).
* install_CPU.sh
  * Same as install.sh, but ANN classifier plugin uses tensorflow without GPU (enabling the GPU parameter in the GUI has no effect). 
  * ./install.sh is preferable choice please use it if you can
* install_editable_CPU.sh
  * Same as install_editable.sh, but ANN classifier plugin uses tensorflow without GPU (enabling the GPU parameter in the GUI has no effect).
  * install_editable.sh is preferable choice please use it if you can

If you want to install ClassMark on others systems (eg. Windows), than please take a look at install*.sh scripts to get inspiration. Their just consists of multiple pip3 commands and from their content is immediately clear what you need to install.

At the end it is suitable to restart your computer. After restart finishes you should be able to run ClassMark with:

	classmark
	
## Examples
Examples folder contains examples of data set files.