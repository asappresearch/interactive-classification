;function plot_utterances(df) {
    var Xmin = d3.min(df, function (d) {
            return d['x']
        }),
        Xmax = d3.max(df, function (d) {
            return d['x']
        }),
        Ymin = d3.min(df, function (d) {
            return d['y']
        }),
        Ymax = d3.max(df, function (d) {
            return d['y']
        }),
        x = d3.scaleLinear().range([0, width]).nice().domain([Xmin, Xmax]),
        y = d3.scaleLinear().range([height, 0]).nice().domain([Ymin, Ymax]),
        x2 = d3.scaleLinear().range([0, width]).nice(),
        y2 = d3.scaleLinear().range([height, 0]).nice(),
        tip = d3.tip().attr('class', 'd3-tip').html(function (d) {
            return d.utterance + '<br />' + d.intent + '<br />' + d.cluster_id;
        }),
        color = d3.scaleOrdinal(d3.schemeCategory10);

    x2.domain(x.domain());
    y2.domain(y.domain());

    var xAxis = d3.axisBottom(x);
    var yAxis = d3.axisLeft(y);

    var zoomy = d3.zoom()
        .scaleExtent([0.1, Infinity])
        .on("zoom", zoom);

    function zoom() {
        var t = d3.event.transform;
        x.domain(t.rescaleX(x2).domain());
        y.domain(t.rescaleY(y2).domain());
        svg.select(".x.axis").call(xAxis);
        svg.select(".y.axis").call(yAxis);

        window.t = t;
        svg.selectAll(".dot")
            .attr("transform", t)
            .attr("r", 5 / t.k);
    }

    d3.select("#map").selectAll("*").remove();
    var svg = d3.select("#map")
        .append("svg")
        .attr('viewbox', '0 0 ' + width + ' ' + height)
        .attr('width', width)
        .attr('height', height);


    svg.call(tip);

    var map = svg
        .append("g")
        .classed("map-container", true)
        .attr("transform", "translate(" + 50 + "," + 50 + ")");
    map
        .call(zoomy);

    map.append("rect").attr("width", width).attr("height", height);
    map.append("g")
        .classed("x axis", true)
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    map.append("g")
        .classed("y axis", true)
        .call(yAxis);

    var objects = map.append("svg")
        .classed("objects", true)
        .attr("width", width)
        .attr("height", height);

    objects.selectAll(".dot")
        .data(df)
        .enter()
        .append("circle")
        .attr("class", "dot")
        .attr("r", 5)
        .attr("cx", function (d) {
            return (x(d['x']))
        })
        .attr("cy", function (d) {
            return (y(d['y']))
        })
        .style("fill", function (d) {
            return color(d['intent']);
        })
        .on('mouseover', tip.show)
        .on('mouseout', tip.hide);

    var utterances = d3.select("#utterances"),
        utteranceList = utterances.append("ul");
    utteranceList
        .selectAll(".utterance")
        .data(df)
        .enter()
        .append("li")
        .text(function (d) {
            return (d.utterance)
        });
}

function build_treemap(data) {
    var svg = d3.select("#treemap"),
        width = 1000,
        height = 600,
        color = d3.scaleOrdinal(d3.schemeCategory10),
        format = d3.format(",d");

    svg.selectAll("*").remove();

    var treemap = d3.treemap()
        .tile(d3.treemapResquarify)
        .size([width, height])
        .round(true)
        .paddingInner(1);

    var df = d3.nest()
        .key(function (d) {
            return d.intent;
        })
        .entries(data);

    var root = d3.hierarchy({'values': df}, function (x) {
        return x.values
    }).sum(function (x) {
            return x.count
        })
        .sort(function (a, b) {
            return b.value - a.value;
        });

    treemap(root);

    var cell = svg.selectAll("g")
        .data(root.leaves())
        .enter().append("g")
        .attr("transform", function (d) {
            return "translate(" + d.x0 + "," + d.y0 + ")";
        });
    cell.append("rect")
        .attr("id", function (d) {
            return d.data.cluster_id;
        })
        .attr("width", function (d) {
            return d.x1 - d.x0;
        })
        .attr("height", function (d) {
            return d.y1 - d.y0;
        })
        .attr("fill", function (d) {
            return color(d.data.intent);
        })
        .on("click", function (d) {
            window.location = "/cluster/" + d.data.cluster_id;
        });
    cell.append("clipPath")
        .attr("id", function (d) {
            return "clip-" + d.data.cluster_id;
        })
        .append("use")
        .attr("xlink:href", function (d) {
            return "/cluster/" + d.data.cluster_id;
        });
    cell.append("title")
        .text(function (d) {
            return d.data.intent + "\n" + format(d.data.cluster_id);
        });
    cell.append("text")
        .attr("clip-path", function (d) {
            return "url(/cluster/" + d.data.cluster_id + ")";
        })
        .selectAll("tspan")
        .data(function (d) {
            return [d.data.cluster_id]
        })
        .enter()
        .append("tspan")
        .attr("x", 10)
        .attr("y", function (d, i) {
            return 20 + i * 10;
        })
        .text(function (d) {
            return d;
        });
}

function build_trending_table(data){
    var div = d3.select("#trending"),
        table = div.append("table"),
        header = table.append("thead").append("tr"),
        body = table.append("tbody"),
        colors = d3.scaleSequential(d3.interpolateRdYlGn);

    header.append('td').text('Trending Intents');
    table.attr('class', 'table');

    body
        .selectAll("tr")
        .data(data)
        .enter()
        .append('tr')
        .html(function(d){
            var c = colors(d.z_score / 2);
            console.log(c);
           return '<td style="background-color: ' + c + '">' + d.intent + '<span class="sub">Current: ' + d.baseline_avg_freq + ' // Previous: ' + d.prev_avg + '</span>' + '</td>'
        });
}