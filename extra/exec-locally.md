# Executar Localmente

É necessário `ruby` e `jekyll` instalados

**Instalar dependências**
````sh
$ bundle install
````
**Executar**
````sh
bundle exec jekyll serve
````

Vai ler os arquivos e gerar o *build* de arquivos estáticos na pasta `_site/`

Abre em `localhost:4000`

**Observações**
+ Os arquivos `.md` são convertidos em páginas `.html`
  - README vira `index.html`
  - Exemplo
	* O arquivo: `data-sciene/python-BR.md`
	  + Vira a url :`localhost::4000/data-sciene/python-BR.md`

Laout Baseado no Rep: https://github.com/alexeygrigorev/data-science-interviews

**Pode-se editar o título das páginas colocando**

````markdown
---
layout: default
title: Another page
description: This is just another page
---
````

**Links no Markdown**

````markdown
[Link to another page](./another-page.html).
````

**Seção de Colapso (PErguntas e respostas)**

````markdown
# A collapsible section with markdown
<details>
  <summary>Click to expand!</summary>
  
  ## Heading
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>
````