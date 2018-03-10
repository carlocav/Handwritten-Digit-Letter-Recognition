library(png)
library(shinyjs)
library(EBImage)
source('scripts/tools.R', local = TRUE)
source('scripts/neuralnet.R', local = TRUE)

ui <- fluidPage(title="Handwritten Digit and Letter Recognizer",
  tags$head(
    tags$style(HTML("
                    pre, table.table {
                    font-size: smaller;
                    }
                    "))
    ),
  tags$head(tags$script(src="jquery-latest.js")),
  tags$head(tags$script(src="sketch.min.js")),
  useShinyjs(),
  extendShinyjs("scripts/tools.js"),

  fluidRow(
    h1("Handwritten Digit and Letter Recognizer", align = "center"),
    column(offset = 3, width = 6,
    p("capable of distinguishing between digits (0-9), uppercase letters (A-Z) and some lowercase letters that can not be confused with uppercase counterparts (a, b, d, e, f, g, h, n, q, r, t)", style="font-size:160%;", align = "center")
    ),
    br(),
    br(),
    column(offset = 2, width = 5,
           br(),
           fluidRow(
              tags$div(class="tools",
                      tags$a(href='#tools_sketch', "data-tool"='marker', tags$i(class = "glyphicon glyphicon-pencil"), "Marker", class = "btn btn-default action-button", style = "fontweight:600"),
                      tags$a(href='#tools_sketch', "data-tool"='eraser', tags$i(class = "glyphicon glyphicon-erase"), "Eraser", class = "btn btn-default action-button", style = "fontweight:600"))
           ),
           fluidRow(
              tags$div(style="border: 2px solid #9999BB; width: 392px; height:392px;",
                      tags$canvas(id='tools_sketch', width='392px', height='392px'))
           ),
           fluidRow(
              tags$div(class="tools",
                      tags$a(href='#tools_sketch', "data-size"='8', "8"),
                      tags$a(href='#tools_sketch', "data-size"='12', "12"),
                      tags$a(href='#tools_sketch', "data-size"='16', "16"),
                      tags$a(href='#tools_sketch', "data-size"='24', "24"),
                      tags$a(href='#tools_sketch', "data-size"='32', "32"),
                      tags$a(href='#tools_sketch', "data-size"='48', "48"))
           )
    ),
    column(width = 5,
           br(),
           br(),
           br(),
           fluidRow(align="left",
                    actionButton("exportButton", h4("Recognize"))
           ),
           br(),
           br(),
           fluidRow(align="left",
                    plotOutput("plotLowResImage", width = "200px", height = "200px")
           ),
           br(),
           fluidRow(align="left",
                    h3(textOutput("textPredict"))
           )
      )
    ),
    tags$script(type="text/javascript", "$(function() {
                          $('#tools_sketch').sketch({defaultColor: '#000', defaultSize: '20'});
              });
              ")
    )


server <- function(input, output, session) {
  sessionEnvironment = environment()

  onclick("exportButton", {
    print("Load draw")
    js$exportCanvas()
  })

  reactExportPNG <- reactive({
    if (is.null(input$jstextPNG) || nchar(input$jstextPNG)<=0) return()
    data = tool.htmlImageToBlob(input$jstextPNG)
    tool.writeBlob("data/exported.png", data)
    lastRandomRow <<- -1
  })
  observe(reactExportPNG())

  output$plotLowResImage <- renderPlot({
    reactExportPNG()
    m = tool.readPNG("data/exported.png")  # mSmall
    if (is.null(m) || nrow(m)<=0) return()
    m = m[1:nrow(m),ncol(m):1]    #rotation
    par.default = par()$mar
    par(mar = c(0.14, 0, 0, 0))
    plot.new()
    image(1-m, col= grey.colors(100, 0, 1, gamma = 0.5), add=TRUE)
    box()
    par(mar = par.default)
  })

  output$textPredict <- renderText({
    reactExportPNG()
    m = tool.readPNG("data/exported.png")   # mSmall
    m = matrix(t(m), nrow=1)
    print(m)
    paste0("Output: ", predict(m))
  })
}

shinyApp(ui, server)
